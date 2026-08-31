#lang racket/base
;; Generate and build the custom C++ shim for the few kernels the stable
;; ABI and the generated per-backend shims do not cover. Bodies are
;; hand-curated (checked against ATen declarations); the build needs the
;; venv torch headers and a C++ toolchain at build time only. Output:
;; build/zl_shim.cpp and build/libzl_shim.dylib.

(require racket/file
         racket/runtime-path
         racket/string
         racket/system)

(define-runtime-path repo-root "..")

(define shim-source #<<EOF
#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

using torch::aot_inductor::new_tensor_handle;
using torch::aot_inductor::tensor_handle_to_tensor_pointer;

static inline at::Tensor& T(AtenTensorHandle h) {
  return *tensor_handle_to_tensor_pointer(h);
}

extern "C" {

AOTITorchError zl_sub(AtenTensorHandle a, AtenTensorHandle b, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::sub(T(a), T(b)));
  });
}

AOTITorchError zl_div(AtenTensorHandle a, AtenTensorHandle b, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::div(T(a), T(b)));
  });
}

AOTITorchError zl_neg(AtenTensorHandle x, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::neg(T(x)));
  });
}

AOTITorchError zl_rsqrt(AtenTensorHandle x, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::rsqrt(T(x)));
  });
}

AOTITorchError zl_silu(AtenTensorHandle x, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::silu(T(x)));
  });
}

AOTITorchError zl_mean(AtenTensorHandle x,
                       const int64_t* dims,
                       int64_t ndims,
                       int32_t keepdim,
                       AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(
        at::mean(T(x), c10::IntArrayRef(dims, ndims), keepdim != 0));
  });
}

AOTITorchError zl_softmax(AtenTensorHandle x, int64_t dim, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::softmax(T(x), dim));
  });
}

AOTITorchError zl_embedding(AtenTensorHandle weight,
                            AtenTensorHandle indices,
                            AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::embedding(T(weight), T(indices)));
  });
}

AOTITorchError zl_argmax(AtenTensorHandle x,
                         int64_t dim,
                         int32_t keepdim,
                         AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(at::argmax(T(x), dim, keepdim != 0));
  });
}

} // extern "C"
EOF
  )

(define (find-torch-dir)
  (define venv-lib (build-path repo-root ".venv/lib"))
  (unless (directory-exists? venv-lib)
    (error 'gen-wrapper "no .venv found; run tools/setup-python.sh first"))
  (or (for/or ([d (in-list (directory-list venv-lib))])
        (define p (build-path venv-lib d "site-packages/torch"))
        (and (regexp-match? #rx"^python3" (path->string d))
             (directory-exists? p)
             p))
      (error 'gen-wrapper "torch not found in .venv; run tools/setup-python.sh")))

(module+ main
  (define torch-dir (find-torch-dir))
  (define build-dir (build-path repo-root "build"))
  (make-directory* build-dir)
  (define cpp (build-path build-dir "zl_shim.cpp"))
  (define dylib (build-path build-dir "libzl_shim.dylib"))
  (call-with-output-file cpp #:exists 'replace
    (lambda (out) (display shim-source out)))
  (define args
    (list "-std=c++17"
          "-O1"
          "-fPIC"
          "-shared"
          (path->string cpp)
          "-o"
          (path->string dylib)
          (string-append "-I" (path->string (build-path torch-dir "include")))
          (string-append "-I"
                         (path->string (build-path torch-dir "include/torch/csrc/api/include")))
          (string-append "-L" (path->string (build-path torch-dir "lib")))
          "-ltorch_cpu"
          "-lc10"
          (string-append "-Wl,-rpath," (path->string (build-path torch-dir "lib")))))
  (printf "clang++ ~a\n" (string-join args " "))
  (unless (apply system* "/usr/bin/clang++" args)
    (error 'gen-wrapper "shim compilation failed"))
  (printf "shim ready: ~a\n" dylib))
