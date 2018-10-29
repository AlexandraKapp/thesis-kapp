#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
//#include <io.h>
#include <fcntl.h>
//#include <sys/mman.h>
#include "list.h"

#define VERSION "0.2"
//#define DEBUG

unsigned long long corr_count = 0;

// compute pearson correlation coefficient between time series at positions i1 and i2 in d (of length l)
// NOTE: result may be nan, if the variance of any of the time series is zero, or if
// any of the time series contains nans
double pearson2(const double *d, const unsigned long long i1, const unsigned long long i2, const unsigned long long l) {
  unsigned int i;
  double mean1, mean2, var1, var2, cov;

  // compute means
  mean1 = 0.0; mean2 = 0.0;
  for (i = 0; i < l; i++) {
    mean1 += d[i1*l+i]; mean2 += d[i2*l+i];
  }
  mean1 /= l; mean2 /= l;

  // compute variances and covariance
  var1 = 0.0; var2 = 0.0;
  cov = 0.0;
  for (i = 0; i < l; i++) {
    var1 += (d[i1*l+i]-mean1)*(d[i1*l+i]-mean1);
    var2 += (d[i2*l+i]-mean2)*(d[i2*l+i]-mean2);
    cov += (d[i1*l+i]-mean1)*(d[i2*l+i]-mean2);
  }
  var1 /= (l-1); var2 /= (l-1); cov /= (l-1); // denominators don't really matter

  // compute correlation
  cov = cov/(sqrt(var1)*sqrt(var2));
  return cov > 1 ? 1 : cov;
}

// compute n-by-n correlation matrix for complete data set d with n rows and l columns
PyArrayObject *
pearson(const double *d, unsigned long long n, unsigned long long l) {
  PyArrayObject *coef;
  long long int dim[2];
  long long int ij, i, j;
  long long n2 = n * (n - 1) / 2;
  double p;

  dim[0] = n; dim[1] = n;
  coef = (PyArrayObject *) PyArray_ZEROS(2, dim, NPY_DOUBLE, 0);
  if(!coef) {
    PyErr_SetString(PyExc_MemoryError, "Cannot create output array1.");
    return NULL;
  }

  for (i = 0; i < n; i++) {
    (*(double *) PyArray_GETPTR2(coef, i, i)) = 1.0;
  }

#pragma omp parallel for private(i, j, p) if (n2 > 8)
  for (ij = 0; ij < n2; ij++) {
      i = ij/n;
      j = ij%n;
      if (j <= i) {
        i = n - (i + 2);
        j = n - (j + 1);
      }

      p = pearson2(d, i, j, l);
      (*(double *) PyArray_GETPTR2(coef, i, j)) = p;
      (*(double *) PyArray_GETPTR2(coef, j, i)) = p;
  }

  return coef;
}

// compute upper triangular part of the correlation matrix
// and store as a vector of length n*(n+1)/2
//
// original code by Aljoscha Rheinwalt
// adapted by Erik Scharwächter
//
// d: data array with n rows and l columns
// diagonal: (bool) include values on diagonal, default: 0
// mmap_arr: (bool) create temporary memory mapped file to hold the coefficient array (for large data sets)
// mmap_fd: pointer to an uninitialized (!) file descriptor for the mmap array, will be initialized
PyArrayObject *
pearson_triu(const double *d, unsigned long long n, unsigned long long l, int diagonal, int mmap_arr, int *mmap_fd) {
  PyArrayObject *coef;
  double *mmap_data;
  char mmap_filename[] = "tmpTriuCorrMat.XXXXXX";
  long long int dim;
  int errcode;

  long long i, k, o;
  double mk, sk, dk, h;
  double mi, si, sum;
  double *m, *s;
  double *c;

  if (diagonal)
    dim = n * (n + 1) / 2;
  else
    dim = n * (n - 1) / 2;

  if (!mmap_arr) {
    coef = (PyArrayObject *) PyArray_ZEROS(1, &dim, NPY_DOUBLE, 0);
    if(!coef) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create output array2.");
      return NULL;
    }
  } else {
	  PyErr_SetString(PyExc_MemoryError, "Removed MMAP support for windows build /Patrick");
	  return NULL;
  }

  /* mean and std */
  m = malloc(n * sizeof(double));
  s = malloc(n * sizeof(double));
  if (!m || !s) {
    PyErr_SetString(PyExc_MemoryError, "Cannot create mean and std arrays.");
    return NULL;
  }
#pragma omp parallel for private(k, h, mk, sk, dk)
  for (i = 0; i < (long long)n; i++) {
    mk = sk = 0;
    for (k = 0; k < (long long)l; k++) {
      dk = d[i*l + k];
      h = dk - mk;
      mk += h / (k + 1);
      sk += h * (dk - mk);
    }
    m[i] = mk;
    s[i] = sqrt(sk / (l - 1));
  }

  /* dot products */
  c = (double *) PyArray_DATA(coef);
#pragma omp parallel for private(k, mi, si, mk, sk, o, sum)
  for (i = 0; i < (long long)n; i++) {
    mi = m[i];
    si = s[i];
    for (k = i+(1-diagonal); k < (long long)n; k++) {
      mk = m[k];
      sk = s[k];
      sum = 0;
      for (o = 0; o < (long long)l; o++)
        sum += (d[i*l + o] - mi) * (d[k*l + o] - mk) / si / sk;
      if (diagonal)
        c[i*n-i*(i+1)/2+k] = sum / (l - 1);
      else
        c[i*(n-1)-i*(i+1)/2+k-1] = sum / (l - 1);
    }
  }
  free(m);
  free(s);

  return coef;
}

// find equivalence classes in a time series data set
//
// d: data set with n rows (time series) and l columns (time steps)
// alpha: transitivity threshold
// kappa: minimum cluster size
// max_nan: maximum number of nans within a pivot time series
PyArrayObject *
cluster(const double *d, unsigned long long n, unsigned long long l, double alpha, unsigned long long kappa, unsigned long long max_nan)
{
  unsigned long long pivot, i, nan_count;
  double rho;
  llist_ul timeseries_l;
  llist_ul *clustermemb_pos_l;
  llist_ul *clustermemb_neg_l;
  llist_ul *noise_l;
  llist_ptr cluster_l;
  llist_item_ul *iter_ul, *iter_ul_next;
  llist_item_ptr *iter_ptr;

  corr_count = 0;

  PyArrayObject *membs = (PyArrayObject *) PyArray_ZEROS(1, (unsigned long long int *) &n, NPY_LONGLONG, 0);
  if(!membs) {
      PyErr_SetString(PyExc_MemoryError, "Cannot create output array3.");
      return NULL;
  }

  // initialize time series index list
  llist_ul_init(&timeseries_l);
  for (i = 0; i < n; i++) {
    llist_ul_push_back(&timeseries_l, i);
  }

  // initialize cluster list
  llist_ptr_init(&cluster_l);

  // initialize noise cluster and add to cluster list (always at position 1)
  noise_l = (llist_ul *) malloc(sizeof(llist_ul));
  if (!noise_l) return NULL;
  llist_ul_init(noise_l);
  llist_ptr_push_back(&cluster_l, noise_l);

  // iterate over all time series until none is left
  while (llist_ul_size(&timeseries_l) > 0) {
 #ifdef DEBUG
    printf("\r% 9lld left...", llist_ul_size(&timeseries_l));
 #endif
    pivot = llist_ul_back(&timeseries_l);

    // check if pivot contains too many nans to be considered a pivot
    nan_count = 0;
    for (i = 0; i < l; i++) {
      if (isnan(d[pivot*l+i])) nan_count++;
    }
    if (nan_count > max_nan) {
      // add pivot to noise cluster
      printf("pivot %lld has too many nans\n", pivot);
      llist_ul_relink(timeseries_l.last, &timeseries_l, noise_l);
      continue;
    }

    // initialize positive and negative clusters
    clustermemb_pos_l = (llist_ul *) malloc(sizeof(llist_ul));
    if (!clustermemb_pos_l) return NULL;
    llist_ul_init(clustermemb_pos_l);
    clustermemb_neg_l = (llist_ul *) malloc(sizeof(llist_ul));
    if (!clustermemb_neg_l) return NULL;
    llist_ul_init(clustermemb_neg_l);

    // compute all correlations between pivot and remaining time series
    // and create positive and negative protoclusters
    iter_ul = timeseries_l.first;
    while (iter_ul != NULL) {
      iter_ul_next = iter_ul->next; // store successor before relinking
      corr_count++;
      rho = pearson2(d, pivot, iter_ul->data, l);
      if (isnan(rho)) {
        // TODO: we add the tested time series to the noise cluster, this might not be
        // a good idea if nan value occurs because there are no overlapping valid time steps
        // in pivot and tested time series
        //printf("rho=nan for pivot %lld and time series %lld\n", pivot, iter_ul->data);
        llist_ul_relink(iter_ul, &timeseries_l, noise_l);
      } else {
        if (rho >=  alpha) llist_ul_relink(iter_ul, &timeseries_l, clustermemb_pos_l);
        if (rho <= -alpha) llist_ul_relink(iter_ul, &timeseries_l, clustermemb_neg_l);
      }
      iter_ul = iter_ul_next;
    }

#ifdef DEBUG
    llist_item_ul *tmp_iter1;
    llist_item_ul *tmp_iter2;
    tmp_iter1 = clustermemb_pos_l->first;
    while (tmp_iter1) {
      tmp_iter2 = tmp_iter1;
      while (tmp_iter2) {
        printf("pos %lld %lld %.2f\n", tmp_iter1->data, tmp_iter2->data, pearson2(d, tmp_iter1->data, tmp_iter2->data, l));
        tmp_iter2 = tmp_iter2->next;
      }
      tmp_iter1 = tmp_iter1->next;
    }
    tmp_iter1 = clustermemb_neg_l->first;
    while (tmp_iter1) {
      tmp_iter2 = tmp_iter1;
      while (tmp_iter2) {
        printf("neg %lld %lld %.2f\n", tmp_iter1->data, tmp_iter2->data, pearson2(d, tmp_iter1->data, tmp_iter2->data, l));
        tmp_iter2 = tmp_iter2->next;
      }
      tmp_iter1 = tmp_iter1->next;
    }
#endif

    // check whether protoclusters fulfill the minimium size constraints
    if (llist_ul_size(clustermemb_pos_l) >= kappa) {
      // add to final clustering
#ifdef DEBUG
      printf("A\n");
#endif
      llist_ptr_push_back(&cluster_l, clustermemb_pos_l);
    } else {
      // relink all time series to noise cluster
#ifdef DEBUG
      printf("B\n");
#endif
      llist_ul_relink_all(clustermemb_pos_l, noise_l);
      free(clustermemb_pos_l);
    }
    if (llist_ul_size(clustermemb_neg_l) >= kappa) {
      // add to final clustering
#ifdef DEBUG
      printf("C\n");
#endif
      llist_ptr_push_back(&cluster_l, clustermemb_neg_l);
    } else {
#ifdef DEBUG
      printf("D\n");
#endif
      // relink all time series to noise cluster
      llist_ul_relink_all(clustermemb_neg_l, noise_l);
      free(clustermemb_neg_l);
    }
  }
#ifdef DEBUG
  printf("\rclustering finished\n");
#endif

  // prepare output array with cluster assignments
  // skip noise cluster (membs id=0 during initialization)
  i = 1;
  iter_ptr = cluster_l.first->next;
  while (iter_ptr != NULL) {
    iter_ul = ((llist_ul *) iter_ptr->data)->first;
    while (iter_ul != NULL) {
      (*(long long int *) PyArray_GETPTR1(membs, iter_ul->data)) = i;
#ifdef DEBUG
      printf("%lld -> %lld\n", iter_ul->data, i);
#endif
      iter_ul = iter_ul->next;
    }
    llist_ul_destroy((llist_ul *) iter_ptr->data);
    free(iter_ptr->data);
    iter_ptr = iter_ptr->next;
    i++;
  }
  llist_ptr_destroy(&cluster_l);
  llist_ul_destroy(&timeseries_l);

  return membs;
}


/* ######################## PYTHON BINDINGS ######################## */


static PyObject *
BlockCorr_Pearson2(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data;
  int i, j;
  double coef;

  if(!PyArg_ParseTuple(args, "Oii", &arg, &i, &j))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  coef = pearson2((double *)PyArray_DATA(data), i, j, PyArray_DIM(data, 1));

  Py_DECREF(data);
  return Py_BuildValue("d", coef);
}

static PyObject *
BlockCorr_Pearson(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef;
  int threads = omp_get_num_procs();

  if(!PyArg_ParseTuple(args, "O|i", &arg, &threads))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  omp_set_num_threads(threads);

  coef = pearson((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1));

  Py_DECREF(data);
  return PyArray_Return(coef);
}

/* TODO: mmap_fd is never closed and file is forgotten -> unnecessary hdd consumption */
static PyObject *
BlockCorr_PearsonTriu(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *coef;
  int diagonal, mmap_arr;
  int mmap_fd;

  diagonal = 0;
  mmap_arr = 0;
  if(!PyArg_ParseTuple(args, "O|ii", &arg, &diagonal, &mmap_arr))
    return NULL;
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  coef = pearson_triu((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1),
      diagonal, mmap_arr, &mmap_fd);

  Py_DECREF(data);
  return PyArray_Return(coef);
}

static PyObject *
BlockCorr_Cluster(PyObject *self, PyObject* args) {
  PyObject *arg;
  PyArrayObject *data, *clus;
  double alpha;
  unsigned long long kappa, max_nan;

  if(!PyArg_ParseTuple(args, "OdKK", &arg, &alpha, &kappa, &max_nan))
    return NULL;
  //printf("%lf %lld \n", alpha, kappa);
  data = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
    NPY_DOUBLE, 2, 2);
  if(!data)
    return NULL;

  clus = cluster((double *)PyArray_DATA(data), PyArray_DIM(data, 0), PyArray_DIM(data, 1),
      alpha, kappa, max_nan);

  Py_DECREF(data);
  return PyArray_Return(clus);
}

static PyObject *
CorrCount(PyObject *self, PyObject* args) {
  return PyLong_FromLongLong(corr_count);
}

static PyMethodDef BlockCorr_methods[] = {
  {"CorrCount", CorrCount, METH_NOARGS,
   "number of corr calcs in Cluser so far\n"},
  {"Pearson2", BlockCorr_Pearson2, METH_VARARGS,
   "corr = Pearson2(data, i, j)\n\n... for a single pair in data\n"},
  {"Pearson", BlockCorr_Pearson, METH_VARARGS,
   "corr = Pearson(data, threads=-1)\n\n...\n"},
  {"PearsonTriu", BlockCorr_PearsonTriu, METH_VARARGS,
   "triu_corr = PearsonTriu(data, diagonal=False, mmap=0)\n\nReturn Pearson product-moment correlation coefficients.\n\nParameters\n----------\ndata : array_like\nA 2-D array containing multiple variables and observations. Each row of `data` represents a variable, and each column a single observation of all those variables.\n\nReturns\n-------\ntriu_corr : ndarray\nThe upper triangle of the correlation coefficient matrix of the variables.\n"},
  {"Cluster", BlockCorr_Cluster, METH_VARARGS,
   "labels = Cluster(data, alpha, kappa, max_nan)\n\n...\n"},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_INIT(name) void init##name(void)
  #define MOD_SUCCESS_VAL(val)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(BlockCorr)
{
  PyObject *m;
  MOD_DEF(m, "BlockCorr", "Block matrix estimation for correlation coefficients.",
          BlockCorr_methods)
  if (m == NULL)
    return MOD_SUCCESS_VAL(-1);
  import_array(); // numpy import
  return MOD_SUCCESS_VAL(m);
}

int
main(int argc, char **argv) {
  Py_SetProgramName(argv[0]);
  Py_Initialize();
    PyImport_ImportModule("BlockCorr");
  Py_Exit(0);
  return 0;
}
