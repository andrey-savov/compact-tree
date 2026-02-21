/*
 * _marisa_ext.c -- Fast C trie-index for MarisaTrie
 *
 * Exposes a single Python type: TrieIndex
 *
 *   trie = TrieIndex(label_bytes, label_off, label_len, is_terminal,
 *                    ch_start, ch_cnt, ch_first_byte, ch_node_id,
 *                    ch_pfx_count, n_nodes, total_n, root_is_terminal)
 *
 *   idx = trie.lookup(word)   # raises KeyError on miss
 *
 * All array arguments are raw little-endian bytes
 * (uint32_t arrays except is_terminal and ch_first_byte which are uint8_t).
 *
 * Children within each parent are stored sorted by their label's first byte,
 * enabling O(1) binary-search dispatch.  ch_pfx_count[i] holds the correct
 * MPH prefix-sum value for that child (computed in the original child order
 * by the Python builder so that indices match the Python implementation).
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

    PyObject *m = PyModule_Create(&_marisa_ext_module);
    if (!m) return NULL;

    Py_INCREF(&TrieIndexType);
    if (PyModule_AddObject(m, "TrieIndex", (PyObject *)&TrieIndexType) < 0) {
        Py_DECREF(&TrieIndexType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
