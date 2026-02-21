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

    /* Pointers into a single contiguous backing allocation (mem).
     *
     * Packing strategy (each packed field eliminates one extra cache-line
     * fetch per trie-node traversal by co-locating values always used
     * together):
     *
     *   node_label[v]  = (uint64_t)label_len[v] << 32 | label_off[v]
     *     One 8-byte load replaces two 4-byte loads from separate 40 KB
     *     arrays.  Critical-path: bisect → node_id → this load → memcmp.
     *
     *   node_csr[v]    = (uint64_t)(ch_cnt[v] | (is_terminal[v] << 31)) << 32
     *                    | ch_start[v]
     *     Folds is_terminal into the count's high bit, eliminating the
     *     separate is_terminal[] fetch and the `parent < n_nodes` branch.
     *     The virtual-root entry (index n_nodes) always has bit 31 = 0.
     *
     *   child_desc[flat] = (uint64_t)ch_pfx_count[flat] << 32 | ch_node_id[flat]
     *     One 8-byte load after bisect settles `flat`.  pfx_count was
     *     previously a separate 40 KB array read at the same index.
     *
     *   ch_first_byte[] stays a dense uint8 array — bisect scans it
     *   sequentially; packing would halve cache-line density and add misses.
     *
     * Memory layout (single malloc, all uint64 arrays come first so the
     * byte arrays need no alignment padding):
     *   node_label  [n_nodes]         × 8 bytes
     *   node_csr    [n_nodes + 1]     × 8 bytes   (index n_nodes = virtual root)
     *   child_desc  [total_children]  × 8 bytes
     *   ch_first_byte[total_children] × 1 byte
     *   label_bytes [lb_len]          × 1 byte
     */
    const uint64_t *node_label;     /* packed (label_off, label_len)         */
    const uint64_t *node_csr;       /* packed (ch_start, ch_cnt|is_term_bit) */
    const uint64_t *child_desc;     /* packed (ch_node_id, ch_pfx_count)     */
    const uint8_t  *ch_first_byte;  /* dense uint8 for bisect / linear scan  */
    const uint8_t  *label_bytes;    /* flat UTF-8 label bytes                */

    void   *mem;                    /* single malloc backing block           */
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

    /*
     * Allocate one block with uint64 arrays first (naturally aligned at the
     * malloc base), then the byte arrays at the end.
     *
     *   node_label  [n_nodes]        × 8
     *   node_csr    [n_nodes + 1]    × 8
     *   child_desc  [total_children] × 8
     *   ch_first_byte[total_children]× 1
     *   label_bytes [lb_buf.len]     × 1
     */
    size_t nl_bytes  = (size_t)n_nodes * 8;
    size_t csr_bytes = (size_t)(n_nodes + 1) * 8;
    size_t cd_bytes  = (size_t)total_children * 8;
    size_t total_size = nl_bytes + csr_bytes + cd_bytes
                        + (size_t)total_children + (size_t)lb_buf.len;

    void *mem = malloc(total_size > 0 ? total_size : 1);
    if (!mem) {
        PyErr_NoMemory();
        goto release_all;
    }

    uint8_t *p = (uint8_t *)mem;

    /* --- node_label[v] = label_off[v] | (uint64_t)label_len[v] << 32 --- */
    const uint32_t *lo_ptr = (const uint32_t *)lo_buf.buf;
    const uint32_t *ll_ptr = (const uint32_t *)ll_buf.buf;
    uint64_t *nl = (uint64_t *)p;
    for (size_t v = 0; v < (size_t)n_nodes; v++)
        nl[v] = (uint64_t)ll_ptr[v] << 32 | (uint64_t)lo_ptr[v];
    p += nl_bytes;

    /* --- node_csr[v] = ch_start[v] | (uint64_t)(ch_cnt[v] | (is_terminal[v]<<31)) << 32
     *   virtual root (index n_nodes): is_terminal bit = 0 always.       --- */
    const uint32_t *cs_ptr  = (const uint32_t *)cs_buf.buf;
    const uint32_t *cc_ptr  = (const uint32_t *)cc_buf.buf;
    const uint8_t  *it_ptr  = (const uint8_t  *)it_buf.buf;
    uint64_t *nc = (uint64_t *)p;
    for (size_t v = 0; v < (size_t)n_nodes; v++) {
        uint32_t cnt_t = cc_ptr[v] | ((uint32_t)it_ptr[v] << 31);
        nc[v] = (uint64_t)cnt_t << 32 | (uint64_t)cs_ptr[v];
    }
    /* Virtual root: terminal bit = 0 */
    nc[n_nodes] = (uint64_t)cc_ptr[n_nodes] << 32 | (uint64_t)cs_ptr[n_nodes];
    p += csr_bytes;

    /* --- child_desc[flat] = ch_node_id[flat] | (uint64_t)ch_pfx_count[flat] << 32 --- */
    const uint32_t *cni_ptr = (const uint32_t *)cni_buf.buf;
    const uint32_t *cpc_ptr = (const uint32_t *)cpc_buf.buf;
    uint64_t *cd = (uint64_t *)p;
    for (size_t f = 0; f < (size_t)total_children; f++)
        cd[f] = (uint64_t)cpc_ptr[f] << 32 | (uint64_t)cni_ptr[f];
    p += cd_bytes;

    /* --- ch_first_byte: verbatim copy --- */
    memcpy(p, cfb_buf.buf, (size_t)total_children);
    const uint8_t *cfb = p;
    p += total_children;

    /* --- label_bytes: verbatim copy --- */
    memcpy(p, lb_buf.buf, (size_t)lb_buf.len);
    const uint8_t *lbp = p;

    free(self->mem);
    self->mem            = mem;
    self->node_label     = nl;
    self->node_csr       = nc;
    self->child_desc     = cd;
    self->ch_first_byte  = cfb;
    self->label_bytes    = lbp;
    self->n_nodes        = n_nodes;
    self->total_n        = total_n;
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

    const uint64_t *node_label    = self->node_label;
    const uint64_t *node_csr      = self->node_csr;
    const uint64_t *child_desc    = self->child_desc;
    const uint8_t  *ch_first_byte = self->ch_first_byte;
    const uint8_t  *label_bytes   = self->label_bytes;

    while (rem_len > 0) {
        uint8_t first_byte = (uint8_t)rem[0];

        /* Single 8-byte load: ch_start (low32) + (ch_cnt | is_terminal_bit) (high32) */
        uint64_t csr   = node_csr[parent];
        uint32_t start = (uint32_t)csr;
        uint32_t cnt_t = (uint32_t)(csr >> 32);  /* ch_cnt | (is_terminal << 31) */
        uint32_t count = cnt_t & 0x7FFFFFFFu;

        if (count == 0) {
            PyErr_SetObject(PyExc_KeyError, arg);
            return NULL;
        }

        /* Find the child whose label starts with first_byte.
         * Item 5: linear scan for small counts (L0=9, L1=4 nodes are
         * permanently L1-cached; linear scan avoids bisect branch
         * mispredictions).  Binary search for large counts (L2=10000). */
        uint32_t lo;
#define TRIE_LINEAR_THRESHOLD 16u
        if (count <= TRIE_LINEAR_THRESHOLD) {
            const uint8_t *fb = ch_first_byte + start;
            lo = 0;
            while (lo < count && fb[lo] != first_byte) lo++;
            if (lo >= count) {
                PyErr_SetObject(PyExc_KeyError, arg);
                return NULL;
            }
        } else {
            uint32_t blo = 0, bhi = count;
            while (blo < bhi) {
                uint32_t mid = (blo + bhi) >> 1;
                if (ch_first_byte[start + mid] < first_byte)
                    blo = mid + 1;
                else
                    bhi = mid;
            }
            if (blo >= count || ch_first_byte[start + blo] != first_byte) {
                PyErr_SetObject(PyExc_KeyError, arg);
                return NULL;
            }
            lo = blo;
        }
#undef TRIE_LINEAR_THRESHOLD

        /* Multiple children can share the same first UTF-8 byte (e.g. é and ê
         * both begin with 0xC3 in UTF-8).  Scan the entire equal-range of
         * first_byte matches and pick the first candidate whose full label
         * matches rem via memcmp. */
        uint32_t range_end = start + lo + 1;
        while (range_end < start + count &&
               ch_first_byte[range_end] == first_byte)
            range_end++;

        uint32_t child   = 0;
        uint32_t pfx_cnt = 0;
        uint32_t llen    = 0;
        int      found   = 0;

        for (uint32_t j = start + lo; j < range_end; j++) {
            uint64_t desc = child_desc[j];
            uint32_t c    = (uint32_t)desc;
            uint32_t pc   = (uint32_t)(desc >> 32);
            uint64_t nlbl = node_label[c];
            uint32_t loff = (uint32_t)nlbl;
            uint32_t len  = (uint32_t)(nlbl >> 32);
            if ((Py_ssize_t)len <= rem_len &&
                memcmp(rem, label_bytes + loff, (size_t)len) == 0) {
                child   = c;
                pfx_cnt = pc;
                llen    = len;
                found   = 1;
                break;
            }
        }
        if (!found) {
            PyErr_SetObject(PyExc_KeyError, arg);
            return NULL;
        }

        /* Accumulate MPH index.
         * is_terminal[parent] is the high bit of cnt_t — no separate array
         * load, and no `parent < n_nodes` branch (virtual-root entry always
         * has bit 31 = 0). */
        idx += pfx_cnt;
        idx += cnt_t >> 31;

        rem     += llen;
        rem_len -= (Py_ssize_t)llen;
        parent   = child;
    }

    /* Check terminal on final node (high bit of node_csr high32). */
    if (!((uint32_t)(node_csr[parent] >> 32) >> 31)) {
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
    uint32_t n_elbl;              /* len(elbl) == len(vcol)                 */

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

    /* Cross-field validation: every node's child_start/child_count range
     * must lie within the elbl/vcol arrays.  Catches corrupted or hostile
     * serialised CompactTree data before any lookup can reach C pointers. */
    {
        uint32_t n_elbl_v = (uint32_t)(elbl_buf.len / 4);
        const uint32_t *cs_v = (const uint32_t *)cs_buf.buf;
        const uint32_t *cc_v = (const uint32_t *)cc_buf.buf;
        for (uint32_t v = 0; v < n_tree_nodes; v++) {
            uint32_t cnt    = cc_v[v];
            if (cnt > 0) {
                uint32_t cstart = cs_v[v];
                /* cstart is 1-indexed; (cstart-1)+cnt must not exceed n_elbl */
                if (cstart < 1 ||
                    (uint64_t)(cstart - 1) + cnt > (uint64_t)n_elbl_v) {
                    PyErr_SetString(PyExc_ValueError,
                        "TreeIndex: child_start/count out of elbl bounds");
                    goto release_all;
                }
            }
        }
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
    self->n_elbl       = (uint32_t)(elbl_buf.len / 4);

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

    /* Bounds-check node_pos before indexing into child_count/child_start */
    if ((uint32_t)node_pos >= self->n_tree_nodes) {
        PyErr_SetString(PyExc_IndexError,
                        "TreeIndex.get: node_pos out of range");
        return NULL;
    }

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

    /* Bounds-check node_pos */
    if ((uint32_t)node_pos >= self->n_tree_nodes)
        return PyLong_FromLong(-1);

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
/* TreeIndex.get_path(node_pos, key1, key2, ...) -> int | str          */
/* ------------------------------------------------------------------ */
/*
 * Traverses all keys in a single C call, avoiding per-level Python
 * round-trips.  Returns child_pos (int) if the last key resolves to an
 * internal node, or the leaf value string.  Raises KeyError on any miss.
 */
static PyObject *
TreeIndex_get_path(TreeIndexObject *self, PyObject *args)
{
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError,
                        "get_path requires node_pos plus at least one key");
        return NULL;
    }

    unsigned long upos =
        PyLong_AsUnsignedLong(PyTuple_GET_ITEM(args, 0));
    if (upos == (unsigned long)-1 && PyErr_Occurred())
        return NULL;
    uint32_t pos = (uint32_t)upos;

    /* Validate initial pos */
    if (pos >= self->n_tree_nodes) {
        PyErr_SetString(PyExc_IndexError,
                        "TreeIndex.get_path: node_pos out of range");
        return NULL;
    }

    Py_ssize_t last = nargs - 1;
    for (Py_ssize_t i = 1; i <= last; i++) {
        PyObject *key = PyTuple_GET_ITEM(args, i);

        /* key -> key_id */
        uint32_t key_id = tree_key_to_id(self->key_trie, key);
        if (key_id == UINT32_MAX)
            return NULL;   /* KeyError already set */

        /* binary-search children of pos */
        uint32_t count = self->child_count[pos];
        if (count == 0) {
            PyErr_SetObject(PyExc_KeyError, key);
            return NULL;
        }
        uint32_t lo  = self->child_start[pos] - 1;
        uint32_t idx = tree_bisect(self->elbl, lo, lo + count, key_id);
        if (idx == UINT32_MAX) {
            PyErr_SetObject(PyExc_KeyError, key);
            return NULL;
        }
        uint32_t child_pos = idx + 1;

        if (i == last) {
            /* Last key: return leaf string or child_pos for internal nodes */
            uint32_t vv = self->vcol[child_pos - 1];
            if (vv == TREE_INTERNAL)
                return PyLong_FromUnsignedLong(child_pos);
            PyObject *vid_obj = PyLong_FromUnsignedLong(vv);
            if (!vid_obj) return NULL;
            PyObject *result = PyObject_CallOneArg(self->val_restore, vid_obj);
            Py_DECREF(vid_obj);
            return result;
        }

        /* Intermediate key: node must be internal to descend */
        if (self->vcol[child_pos - 1] != TREE_INTERNAL) {
            PyErr_SetObject(PyExc_KeyError, key);
            return NULL;
        }
        pos = child_pos;
    }
    Py_RETURN_NONE;  /* unreachable */
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
    {"get_path", (PyCFunction)TreeIndex_get_path, METH_VARARGS,
     "get_path(node_pos, key1, key2, ...) -> int | str\n\n"
     "Traverse all keys in a single C call.  Returns child_pos (int) for\n"
     "an internal final node, or the leaf value string.  Raises KeyError\n"
     "on any miss."},
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
