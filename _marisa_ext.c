/*
 * _marisa_ext.c -- Fast C extensions for MarisaTrie and CompactTree
 *
 * Exposes two Python types:
 *
 *   TrieIndex  -- MPH index for a single MarisaTrie (word -> int)
 *
 *   TreeIndex  -- CompactTree traversal helper.
 *                 Holds the tree's CSR arrays (elbl / vcol / child_start /
 *                 child_count) and a direct C pointer to the key TrieIndex so
 *                 that get(node_pos, key) performs the full
 *                   key -> key_id -> bisect elbl -> vcol check -> value
 *                 pipeline in one C call, with no intermediate Python frames.
 *
 * TrieIndex constructor:
 *   trie = TrieIndex(label_bytes, label_off, label_len, is_terminal,
 *                    ch_start, ch_cnt, ch_first_byte, ch_node_id,
 *                    ch_pfx_count, n_nodes, total_n, root_is_terminal)
 *   idx = trie.lookup(word)        # raises KeyError on miss
 *
 * TreeIndex constructor:
 *   ti = TreeIndex(elbl, vcol, child_start, child_count,
 *                  n_tree_nodes, key_trie, val_restore)
 *   result = ti.get(node_pos, key)   # int (child_pos) | str (leaf value)
 *   pos    = ti.find(node_pos, key)  # int >= 0 | -1 on miss, no exception
 *
 * All array arguments are raw little-endian bytes (uint32_t arrays except
 * is_terminal and ch_first_byte which are uint8_t).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* TrieIndex object                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject_HEAD
    uint32_t  n_nodes;
    uint32_t  total_n;
    int       root_is_terminal;
    uint32_t  total_children;

    /* Pointers into a single contiguous backing allocation (mem). */
    const uint8_t  *label_bytes;    /* flat UTF-8 label bytes             */
    const uint32_t *label_off;      /* label_bytes offset per node        */
    const uint32_t *label_len;      /* byte length of label per node      */
    const uint8_t  *is_terminal;    /* 1 if terminal, per node            */

    /* CSR children arrays (n_nodes+1 entries; index n_nodes = virtual root) */
    const uint32_t *ch_start;       /* start index in child arrays        */
    const uint32_t *ch_cnt;         /* child count                        */

    /* Flat child arrays (total_children entries, sorted by first_byte/parent) */
    const uint8_t  *ch_first_byte;  /* first byte of child label          */
    const uint32_t *ch_node_id;     /* child node id                      */
    const uint32_t *ch_pfx_count;   /* MPH prefix-sum for this child      */

    void   *mem;                    /* single malloc backing block        */
} TrieIndexObject;

/* ------------------------------------------------------------------ */
/* dealloc                                                              */
/* ------------------------------------------------------------------ */

static void
TrieIndex_dealloc(TrieIndexObject *self)
{
    free(self->mem);
    self->mem = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ------------------------------------------------------------------ */
/* __new__                                                              */
/* ------------------------------------------------------------------ */

static PyObject *
TrieIndex_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TrieIndexObject *self = (TrieIndexObject *)type->tp_alloc(type, 0);
    if (self) {
        self->mem = NULL;
        self->n_nodes = 0;
        self->total_n = 0;
        self->root_is_terminal = 0;
        self->total_children   = 0;
    }
    return (PyObject *)self;
}

/* ------------------------------------------------------------------ */
/* __init__                                                             */
/* ------------------------------------------------------------------ */

static int
TrieIndex_init(TrieIndexObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        "label_bytes", "label_off", "label_len", "is_terminal",
        "ch_start", "ch_cnt", "ch_first_byte", "ch_node_id", "ch_pfx_count",
        "n_nodes", "total_n", "root_is_terminal",
        NULL
    };

    Py_buffer lb_buf, lo_buf, ll_buf, it_buf;
    Py_buffer cs_buf, cc_buf, cfb_buf, cni_buf, cpc_buf;
    unsigned int n_nodes, total_n, root_is_terminal;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            "y*y*y*y*y*y*y*y*y*III", kwlist,
            &lb_buf,  &lo_buf,  &ll_buf,  &it_buf,
            &cs_buf,  &cc_buf,  &cfb_buf, &cni_buf, &cpc_buf,
            &n_nodes, &total_n, &root_is_terminal))
        return -1;

    /* Sizes we expect */
    Py_ssize_t node_u32_bytes  = (Py_ssize_t)n_nodes * 4;
    Py_ssize_t csr_u32_bytes   = (Py_ssize_t)(n_nodes + 1) * 4;

    /* Validate sizes */
    if (lo_buf.len  != node_u32_bytes ||
        ll_buf.len  != node_u32_bytes ||
        it_buf.len  != (Py_ssize_t)n_nodes    ||
        cs_buf.len  != csr_u32_bytes  ||
        cc_buf.len  != csr_u32_bytes  ||
        cni_buf.len != cpc_buf.len    ||
        cfb_buf.len != (Py_ssize_t)(cni_buf.len / 4)) {
        PyErr_SetString(PyExc_ValueError, "TrieIndex: array size mismatch");
        goto release_all;
    }

    uint32_t total_children = (uint32_t)(cni_buf.len / 4);

    /* Compute total size needed and allocate one block */
    Py_ssize_t total_size = (Py_ssize_t)(
        lb_buf.len   +              /* label_bytes   */
        lo_buf.len   +              /* label_off     */
        ll_buf.len   +              /* label_len     */
        it_buf.len   +              /* is_terminal   */
        cs_buf.len   +              /* ch_start      */
        cc_buf.len   +              /* ch_cnt        */
        cfb_buf.len  +              /* ch_first_byte */
        cni_buf.len  +              /* ch_node_id    */
        cpc_buf.len                 /* ch_pfx_count  */
    );

    void *mem = malloc((size_t)total_size > 0 ? (size_t)total_size : 1);
    if (!mem) {
        PyErr_NoMemory();
        goto release_all;
    }

    /* Copy each array into the block and set the pointer */
    uint8_t *p = (uint8_t *)mem;

#define COPY_BUF(dst_ptr, src_pbuf, ctype) \
    memcpy(p, (src_pbuf).buf, (size_t)(src_pbuf).len); \
    (dst_ptr) = (const ctype *)p;                       \
    p += (src_pbuf).len;

    COPY_BUF(self->label_bytes,   lb_buf,  uint8_t)
    COPY_BUF(self->label_off,     lo_buf,  uint32_t)
    COPY_BUF(self->label_len,     ll_buf,  uint32_t)
    COPY_BUF(self->is_terminal,   it_buf,  uint8_t)
    COPY_BUF(self->ch_start,      cs_buf,  uint32_t)
    COPY_BUF(self->ch_cnt,        cc_buf,  uint32_t)
    COPY_BUF(self->ch_first_byte, cfb_buf, uint8_t)
    COPY_BUF(self->ch_node_id,    cni_buf, uint32_t)
    COPY_BUF(self->ch_pfx_count,  cpc_buf, uint32_t)

#undef COPY_BUF

    free(self->mem);  /* in case re-init is called */
    self->mem              = mem;
    self->n_nodes          = n_nodes;
    self->total_n          = total_n;
    self->root_is_terminal = (int)root_is_terminal;
    self->total_children   = total_children;

    PyBuffer_Release(&lb_buf);  PyBuffer_Release(&lo_buf);
    PyBuffer_Release(&ll_buf);  PyBuffer_Release(&it_buf);
    PyBuffer_Release(&cs_buf);  PyBuffer_Release(&cc_buf);
    PyBuffer_Release(&cfb_buf); PyBuffer_Release(&cni_buf);
    PyBuffer_Release(&cpc_buf);
    return 0;

release_all:
    PyBuffer_Release(&lb_buf);  PyBuffer_Release(&lo_buf);
    PyBuffer_Release(&ll_buf);  PyBuffer_Release(&it_buf);
    PyBuffer_Release(&cs_buf);  PyBuffer_Release(&cc_buf);
    PyBuffer_Release(&cfb_buf); PyBuffer_Release(&cni_buf);
    PyBuffer_Release(&cpc_buf);
    return -1;
}

/* ------------------------------------------------------------------ */
/* lookup(word) -> int                                                  */
/* ------------------------------------------------------------------ */

static PyObject *
TrieIndex_lookup(TrieIndexObject *self, PyObject *arg)
{
    if (!PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "expected str");
        return NULL;
    }

    Py_ssize_t word_len;
    const char *word = PyUnicode_AsUTF8AndSize(arg, &word_len);
    if (!word) return NULL;

    if (self->total_n == 0) {
        PyErr_SetObject(PyExc_KeyError, arg);
        return NULL;
    }

    /* Empty string */
    if (word_len == 0) {
        if (self->root_is_terminal)
            return PyLong_FromLong(0);
        PyErr_SetObject(PyExc_KeyError, arg);
        return NULL;
    }

    uint32_t idx    = self->root_is_terminal ? 1 : 0;
    uint32_t parent = self->n_nodes;   /* virtual root */
    const char   *rem     = word;
    Py_ssize_t    rem_len = word_len;

    const uint8_t  *label_bytes   = self->label_bytes;
    const uint32_t *label_off     = self->label_off;
    const uint32_t *label_len     = self->label_len;
    const uint8_t  *is_terminal   = self->is_terminal;
    const uint32_t *ch_start      = self->ch_start;
    const uint32_t *ch_cnt        = self->ch_cnt;
    const uint8_t  *ch_first_byte = self->ch_first_byte;
    const uint32_t *ch_node_id    = self->ch_node_id;
    const uint32_t *ch_pfx_count  = self->ch_pfx_count;
    uint32_t        n_nodes       = self->n_nodes;

    while (rem_len > 0) {
        uint8_t  first_byte = (uint8_t)rem[0];
        uint32_t start      = ch_start[parent];
        uint32_t count      = ch_cnt[parent];

        if (count == 0) {
            PyErr_SetObject(PyExc_KeyError, arg);
            return NULL;
        }

        /* Binary search for first_byte within [start, start+count) */
        uint32_t lo = 0, hi = count;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (ch_first_byte[start + mid] < first_byte)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo >= count || ch_first_byte[start + lo] != first_byte) {
            PyErr_SetObject(PyExc_KeyError, arg);
            return NULL;
        }

        uint32_t flat  = start + lo;
        uint32_t child = ch_node_id[flat];
        uint32_t llen  = label_len[child];

        /* Verify full label matches */
        if ((Py_ssize_t)llen > rem_len ||
            memcmp(rem, label_bytes + label_off[child], (size_t)llen) != 0) {
            PyErr_SetObject(PyExc_KeyError, arg);
            return NULL;
        }

        /* Accumulate MPH index */
        idx += ch_pfx_count[flat];
        if (parent < n_nodes && is_terminal[parent])
            idx += 1;

        rem     += llen;
        rem_len -= (Py_ssize_t)llen;
        parent   = child;
    }

    if (!is_terminal[parent]) {
        PyErr_SetObject(PyExc_KeyError, arg);
        return NULL;
    }

    return PyLong_FromUnsignedLong(idx);
}

/* ------------------------------------------------------------------ */
/* Method table                                                         */
/* ------------------------------------------------------------------ */

static PyMethodDef TrieIndex_methods[] = {
    {"lookup", (PyCFunction)TrieIndex_lookup, METH_O,
     "lookup(word) -> int\n\nReturn the MPH index for word. Raises KeyError on miss."},
    {NULL, NULL, 0, NULL}
};

/* ------------------------------------------------------------------ */
/* Type definition                                                      */
/* ------------------------------------------------------------------ */

static PyTypeObject TrieIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_marisa_ext.TrieIndex",
    .tp_basicsize = sizeof(TrieIndexObject),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)TrieIndex_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "C-level radix trie index for fast word-to-int lookup.",
    .tp_methods   = TrieIndex_methods,
    .tp_new       = TrieIndex_new,
    .tp_init      = (initproc)TrieIndex_init,
};

/* ================================================================== */
/* TreeIndex -- CompactTree traversal helper                           */
/* ================================================================== */

/*
 * TreeIndex holds the four CSR arrays of a CompactTree (elbl, vcol,
 * child_start, child_count) plus a direct C pointer to the TrieIndexObject
 * used as the key vocabulary.  This lets get(node_pos, key) collapse the
 * three-step Python hot-path
 *
 *   key_trie[key]          -> key_id
 *   bisect(elbl, key_id)   -> child_pos
 *   vcol[child_pos]        -> leaf_value / _INTERNAL
 *
 * into a single C call with no intermediate Python frames.
 *
 * Array indexing convention (matches compact_tree.py):
 *   elbl[i]         (0-based) = key_id of node at 1-indexed position i+1
 *   vcol[i]         (0-based) = val_id or 0xFFFFFFFF (_INTERNAL) for node i+1
 *   child_start[v]  (0-based node v) = 1-indexed position of first child
 *   child_count[v]  (0-based node v) = number of children
 *
 * So for node v:
 *   first child's 0-based elbl index = child_start[v] - 1
 *   elbl range = [child_start[v]-1, child_start[v]-1 + child_count[v])
 *   child node pos (1-indexed) = elbl_index + 1
 */

#define TREE_INTERNAL  0xFFFFFFFFu

typedef struct {
    PyObject_HEAD

    /* CompactTree arrays (copies inside self->mem) */
    const uint32_t *elbl;         /* edge label (key_id) per non-root node  */
    const uint32_t *vcol;         /* value col (val_id / _INTERNAL) per node*/
    const uint32_t *child_start;  /* 1-indexed first-child pos per node     */
    const uint32_t *child_count;  /* child count per node                   */
    uint32_t n_tree_nodes;        /* len(child_start) == len(child_count)   */

    /*
     * Direct C pointer into the key TrieIndex so TrieIndex_lookup() can be
     * called as a plain C function, skipping all Python function-call
     * dispatch overhead.
     */
    TrieIndexObject *key_trie;    /* borrowed ref into key_trie_obj         */
    PyObject        *key_trie_obj;/* strong ref keeping key_trie alive      */

    /* val_trie.restore_key — Python callable (lru_cached), called for leaves */
    PyObject *val_restore;

    void *mem;                    /* malloc block for the four arrays       */
} TreeIndexObject;

/* ------------------------------------------------------------------ */
/* TreeIndex dealloc / new                                             */
/* ------------------------------------------------------------------ */

static void
TreeIndex_dealloc(TreeIndexObject *self)
{
    free(self->mem);
    Py_XDECREF(self->key_trie_obj);
    Py_XDECREF(self->val_restore);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
TreeIndex_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    TreeIndexObject *self = (TreeIndexObject *)type->tp_alloc(type, 0);
    if (self) {
        self->mem          = NULL;
        self->key_trie_obj = NULL;
        self->val_restore  = NULL;
        self->key_trie     = NULL;
        self->n_tree_nodes = 0;
    }
    return (PyObject *)self;
}

/* ------------------------------------------------------------------ */
/* TreeIndex __init__                                                  */
/* ------------------------------------------------------------------ */

static int
TreeIndex_init(TreeIndexObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        "elbl", "vcol", "child_start", "child_count",
        "n_tree_nodes", "key_trie", "val_restore",
        NULL
    };

    Py_buffer elbl_buf, vcol_buf, cs_buf, cc_buf;
    unsigned int n_tree_nodes;
    PyObject *key_trie_obj, *val_restore;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
            "y*y*y*y*IOO", kwlist,
            &elbl_buf, &vcol_buf, &cs_buf, &cc_buf,
            &n_tree_nodes, &key_trie_obj, &val_restore))
        return -1;

    /* key_trie must be a TrieIndex */
    if (!PyObject_TypeCheck(key_trie_obj, &TrieIndexType)) {
        PyErr_SetString(PyExc_TypeError,
                        "TreeIndex: key_trie must be a TrieIndex");
        goto release_all;
    }
    if (!PyCallable_Check(val_restore)) {
        PyErr_SetString(PyExc_TypeError,
                        "TreeIndex: val_restore must be callable");
        goto release_all;
    }

    Py_ssize_t node_u32 = (Py_ssize_t)n_tree_nodes * 4;
    if (cs_buf.len != node_u32 || cc_buf.len != node_u32) {
        PyErr_SetString(PyExc_ValueError,
                        "TreeIndex: child_start/child_count size mismatch");
        goto release_all;
    }
    if (elbl_buf.len != vcol_buf.len) {
        PyErr_SetString(PyExc_ValueError,
                        "TreeIndex: elbl/vcol size mismatch");
        goto release_all;
    }

    Py_ssize_t total =
        elbl_buf.len + vcol_buf.len + cs_buf.len + cc_buf.len;

    void *mem = malloc((size_t)(total > 0 ? total : 1));
    if (!mem) {
        PyErr_NoMemory();
        goto release_all;
    }

    uint8_t *p = (uint8_t *)mem;

#define TCOPY(dst, pb_, ctype) \
    memcpy(p, (pb_).buf, (size_t)(pb_).len); \
    (dst) = (const ctype *)p;                \
    p += (pb_).len;

    TCOPY(self->elbl,        elbl_buf, uint32_t)
    TCOPY(self->vcol,        vcol_buf, uint32_t)
    TCOPY(self->child_start, cs_buf,   uint32_t)
    TCOPY(self->child_count, cc_buf,   uint32_t)

#undef TCOPY

    free(self->mem);
    self->mem          = mem;
    self->n_tree_nodes = n_tree_nodes;

    Py_INCREF(key_trie_obj);
    Py_XDECREF(self->key_trie_obj);
    self->key_trie_obj = key_trie_obj;
    self->key_trie     = (TrieIndexObject *)key_trie_obj;

    Py_INCREF(val_restore);
    Py_XDECREF(self->val_restore);
    self->val_restore  = val_restore;

    PyBuffer_Release(&elbl_buf); PyBuffer_Release(&vcol_buf);
    PyBuffer_Release(&cs_buf);   PyBuffer_Release(&cc_buf);
    return 0;

release_all:
    PyBuffer_Release(&elbl_buf); PyBuffer_Release(&vcol_buf);
    PyBuffer_Release(&cs_buf);   PyBuffer_Release(&cc_buf);
    return -1;
}

/* ------------------------------------------------------------------ */
/* TreeIndex internals: key lookup + child bisect                      */
/* ------------------------------------------------------------------ */

/*
 * Look up *key* in the key trie and return its id, or UINT32_MAX on miss
 * (KeyError already set on the Python side when total_n > 0; caller clears
 * it for __contains__ paths).
 */
static inline uint32_t
tree_key_to_id(TrieIndexObject *kt, PyObject *key)
{
    PyObject *res = TrieIndex_lookup(kt, key);
    if (!res)
        return UINT32_MAX;
    uint32_t id = (uint32_t)PyLong_AsUnsignedLong(res);
    Py_DECREF(res);
    return id;
}

/*
 * Binary-search self->elbl in [lo, hi) for key_id.
 * Returns the 0-based elbl index on hit, UINT32_MAX on miss.
 */
static inline uint32_t
tree_bisect(const uint32_t *elbl, uint32_t lo, uint32_t hi, uint32_t key_id)
{
    uint32_t blo = lo, bhi = hi;
    while (blo < bhi) {
        uint32_t mid = (blo + bhi) >> 1;
        if (elbl[mid] < key_id)
            blo = mid + 1;
        else
            bhi = mid;
    }
    if (blo >= hi || elbl[blo] != key_id)
        return UINT32_MAX;
    return blo;
}

/* ------------------------------------------------------------------ */
/* TreeIndex.get(node_pos, key) -> int | str                           */
/* ------------------------------------------------------------------ */
/*
 * For an internal node returns the child's 1-indexed node position (int).
 * For a leaf returns the value string (str).
 * Raises KeyError on miss.
 */
static PyObject *
TreeIndex_get(TreeIndexObject *self, PyObject *args)
{
    unsigned int node_pos;
    PyObject *key;
    if (!PyArg_ParseTuple(args, "IO", &node_pos, &key))
        return NULL;

    /* 1. key -> key_id (C call, no Python dispatch) */
    uint32_t key_id = tree_key_to_id(self->key_trie, key);
    if (key_id == UINT32_MAX)
        return NULL;   /* KeyError already set */

    /* 2. Binary-search elbl for key_id among children of node_pos */
    uint32_t count = self->child_count[node_pos];
    if (count == 0) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }
    uint32_t lo = self->child_start[node_pos] - 1;  /* 0-based elbl index */
    uint32_t idx = tree_bisect(self->elbl, lo, lo + count, key_id);
    if (idx == UINT32_MAX) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    /* 3. child node position (1-indexed) */
    uint32_t child_pos = idx + 1;

    /* 4. Check vcol */
    uint32_t vv = self->vcol[child_pos - 1];
    if (vv == TREE_INTERNAL)
        return PyLong_FromUnsignedLong(child_pos);

    /* 5. Leaf: call val_restore(val_id) */
    PyObject *vid_obj = PyLong_FromUnsignedLong(vv);
    if (!vid_obj) return NULL;
    PyObject *result = PyObject_CallOneArg(self->val_restore, vid_obj);
    Py_DECREF(vid_obj);
    return result;
}

/* ------------------------------------------------------------------ */
/* TreeIndex.find(node_pos, key) -> int                                */
/* ------------------------------------------------------------------ */
/*
 * Returns child_pos (>= 1) on hit, or -1 on miss — never raises
 * KeyError.  Useful for __contains__ to avoid exception overhead.
 */
static PyObject *
TreeIndex_find(TreeIndexObject *self, PyObject *args)
{
    unsigned int node_pos;
    PyObject *key;
    if (!PyArg_ParseTuple(args, "IO", &node_pos, &key))
        return NULL;

    uint32_t key_id = tree_key_to_id(self->key_trie, key);
    if (key_id == UINT32_MAX) {
        PyErr_Clear();
        return PyLong_FromLong(-1);
    }

    uint32_t count = self->child_count[node_pos];
    if (count == 0)
        return PyLong_FromLong(-1);

    uint32_t lo  = self->child_start[node_pos] - 1;
    uint32_t idx = tree_bisect(self->elbl, lo, lo + count, key_id);
    if (idx == UINT32_MAX)
        return PyLong_FromLong(-1);

    return PyLong_FromUnsignedLong(idx + 1);  /* 1-indexed child_pos */
}

/* ------------------------------------------------------------------ */
/* TreeIndex method table + type                                       */
/* ------------------------------------------------------------------ */

static PyMethodDef TreeIndex_methods[] = {
    {"get",  (PyCFunction)TreeIndex_get,  METH_VARARGS,
     "get(node_pos, key) -> int | str\n\n"
     "Look up key among children of node_pos.  Returns child_pos (int) for\n"
     "internal nodes or the leaf value string.  Raises KeyError on miss."},
    {"find", (PyCFunction)TreeIndex_find, METH_VARARGS,
     "find(node_pos, key) -> int\n\n"
     "Like get() but returns -1 on miss instead of raising KeyError."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject TreeIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_marisa_ext.TreeIndex",
    .tp_basicsize = sizeof(TreeIndexObject),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)TreeIndex_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "C-level CompactTree traversal helper.",
    .tp_methods   = TreeIndex_methods,
    .tp_new       = TreeIndex_new,
    .tp_init      = (initproc)TreeIndex_init,
};

/* ------------------------------------------------------------------ */
/* Module                                                               */
/* ------------------------------------------------------------------ */

static PyModuleDef _marisa_ext_module = {
    PyModuleDef_HEAD_INIT,
    "_marisa_ext",
    "Fast C trie index for MarisaTrie",
    -1,
    NULL
};

PyMODINIT_FUNC
PyInit__marisa_ext(void)
{
    if (PyType_Ready(&TrieIndexType) < 0)
        return NULL;
    if (PyType_Ready(&TreeIndexType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&_marisa_ext_module);
    if (!m) return NULL;

    Py_INCREF(&TrieIndexType);
    if (PyModule_AddObject(m, "TrieIndex", (PyObject *)&TrieIndexType) < 0) {
        Py_DECREF(&TrieIndexType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&TreeIndexType);
    if (PyModule_AddObject(m, "TreeIndex", (PyObject *)&TreeIndexType) < 0) {
        Py_DECREF(&TreeIndexType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
