#lang typed/racket
;; safetensors parsing: the eight-byte little-endian header length, the
;; JSON tensor directory, and ranged reads of the raw data section. Pure
;; IO with no FFI; materializing bytes into tensors is the caller's
;; business (eager.rkt provides bytes->tensor).

(require "../shape.rkt"
         "../tensor.rkt")

(provide safetensor?
         safetensor-open
         safetensor-names
         safetensor-member?
         safetensor-shape
         safetensor-dtype
         safetensor-bytes)

(require/typed json [read-json (-> Input-Port Any)])

(struct entry
  ([dtype : Dtype] [shape : Shape] [start : Nonnegative-Integer] [end : Nonnegative-Integer])
  #:transparent)

(struct safetensor
  ([path : Path-String] [data-offset : Nonnegative-Integer]
                        [entries : (Immutable-HashTable Symbol entry)])
  #:transparent)

(: dtype-names (Immutable-HashTable String Dtype))
(define dtype-names
  (hash "F32" 'f32 "F16" 'f16 "BF16" 'bf16 "I64" 'i64 "I32" 'i32 "U8" 'u8 "BOOL" 'bool))

(: json-dtype (-> Any Dtype))
(define (json-dtype v)
  (hash-ref dtype-names
            (assert v string?)
            (lambda () (raise-arguments-error 'safetensor-open "unsupported dtype" "dtype" v))))

(: json-shape (-> Any Shape))
(define (json-shape v)
  (map (lambda (d) (assert d exact-nonnegative-integer?)) (assert v list?)))

(: json-offsets (-> Any (Values Nonnegative-Integer Nonnegative-Integer)))
(define (json-offsets v)
  (match (assert v list?)
    [(list s e) (values (assert s exact-nonnegative-integer?) (assert e exact-nonnegative-integer?))]
    [_ (raise-arguments-error 'safetensor-open "malformed data_offsets" "value" v)]))

(: safetensor-open (-> Path-String safetensor))
(define (safetensor-open path)
  (call-with-input-file
   path
   (lambda ([in : Input-Port])
     (define head (read-bytes 8 in))
     (unless (and (bytes? head) (= 8 (bytes-length head)))
       (raise-arguments-error 'safetensor-open "truncated header" "path" path))
     (define header-len (integer-bytes->integer head #f #f 0 8))
     (define header (read-bytes header-len in))
     (unless (and (bytes? header) (= header-len (bytes-length header)))
       (raise-arguments-error 'safetensor-open "truncated header" "path" path))
     (define directory (cast (read-json (open-input-bytes header)) (Immutable-HashTable Symbol Any)))
     (: entries (Immutable-HashTable Symbol entry))
     (define entries
       (for/fold ([acc
                   :
                   (Immutable-HashTable Symbol entry)
                   (hasheq)])
                 ([(name spec) (in-hash directory)])
         (cond
           [(eq? name '__metadata__) acc]
           [else
            (define h (cast spec (Immutable-HashTable Symbol Any)))
            (define dt (json-dtype (hash-ref h 'dtype)))
            (define shp (json-shape (hash-ref h 'shape)))
            (define-values (s e) (json-offsets (hash-ref h 'data_offsets)))
            (unless (= (- e s) (* (shape-numel shp) (dtype-size dt)))
              (raise-arguments-error 'safetensor-open
                                     "byte range does not match shape"
                                     "tensor"
                                     name
                                     "shape"
                                     shp))
            (hash-set acc name (entry dt shp s e))])))
     (safetensor path (+ 8 header-len) entries))))

(: entry-ref (-> safetensor Symbol entry))
(define (entry-ref st name)
  (hash-ref (safetensor-entries st)
            name
            (lambda () (raise-arguments-error 'safetensor-ref "unknown tensor" "name" name))))

(: safetensor-names (-> safetensor (Listof Symbol)))
(define (safetensor-names st)
  (sort (hash-keys (safetensor-entries st)) symbol<?))

(: safetensor-member? (-> safetensor Symbol Boolean))
(define (safetensor-member? st name)
  (hash-has-key? (safetensor-entries st) name))

(: safetensor-shape (-> safetensor Symbol Shape))
(define (safetensor-shape st name)
  (entry-shape (entry-ref st name)))

(: safetensor-dtype (-> safetensor Symbol Dtype))
(define (safetensor-dtype st name)
  (entry-dtype (entry-ref st name)))

(: safetensor-bytes (-> safetensor Symbol Bytes))
(define (safetensor-bytes st name)
  (define e (entry-ref st name))
  (define len (- (entry-end e) (entry-start e)))
  (call-with-input-file
   (safetensor-path st)
   (lambda ([in : Input-Port])
     (file-position in (+ (safetensor-data-offset st) (entry-start e)))
     (define bs (read-bytes len in))
     (unless (and (bytes? bs) (= len (bytes-length bs)))
       (raise-arguments-error 'safetensor-bytes "truncated tensor data" "name" name))
     bs)))
