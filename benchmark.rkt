#lang racket
(require (prefix-in acc- malt/accelerated-tensors/no-duals-no-overrides))
(require (prefix-in flat- malt/flat-tensors/no-duals-no-overrides))
(require (only-in malt/accelerated-tensors
                  (make-printable acc-make-printable)))
(require (only-in malt/flat-tensors
                  (make-printable flat-make-printable)))

(define stdout (current-output-port))
(define iterations (make-parameter 10))

(define (benchmark-prim1 flat-f acc-f name-f (base-shape '()))
  (printf "Input size, ~a Accelerated CPU, ~a Accelerated Real, ~a Accelerated GC, ~a Flat CPU, ~a Flat Real, ~a Flat GC~n"
          name-f name-f name-f name-f name-f name-f)
  (for ((outer-size (in-range 10 (add1 (* 10 (iterations))) 10)))
    (define input-shape (cons outer-size base-shape))
    (define size (apply * input-shape))
    (define numbers (build-list size (λ _ (floor (* 100 (random))))))
    (define flat-in (flat-reshape input-shape (flat-list->tensor numbers)))
    (define acc-in (acc-reshape input-shape (acc-list->tensor numbers)))
    (define-values (flat-res flat-cpu flat-real flat-gc) (time-apply (λ () (flat-f flat-in)) '()))
    (define-values (acc-res acc-cpu acc-real acc-gc) (time-apply (λ () (acc-f acc-in)) '()))
    #;(printf "Flat Input: ~a~n" (flat-make-printable flat-in))
    #;(printf "Flat Output: ~a~n" (flat-make-printable (list-ref flat-res 0)))
    #;(printf "Accelerated Input: ~a~n" (acc-make-printable acc-in))
    #;(printf "Accelerated Output: ~a~n" (acc-make-printable (list-ref acc-res 0)))
    (printf "~a, ~a, ~a, ~a, ~a, ~a, ~a~n" size acc-cpu acc-real acc-gc flat-cpu flat-real flat-gc)))

(define ns (variable-reference->namespace (#%variable-reference)))

(define (string->procedure s)
  (define sym (string->symbol s))
  (eval sym ns))

(define flat-dot-product
  (λ (w t)
    (flat-sum-ρ
      (flat-*-ρ w t))))

(define acc-dot-product
  (λ (w t)
    (acc-sum-ρ
      (acc-*-ρ w t))))

(define (benchmark-sum)
  (printf "Inner size, sum thread Accelerated CPU, sum thread Accelerated Real, sum thread Accelerated GC, sum thread Flat CPU, sum thread Flat Real, sum thread Flat GC~n")
  (for ((inner-size (in-range 10 (add1 (* 10 (iterations))) 10)))
    (define input-shape (list 1000 inner-size))
    (define size (apply * input-shape))
    (define numbers (build-list size (λ _ (floor (* 100 (random))))))
    (define flat-in (flat-reshape input-shape (flat-list->tensor numbers)))
    (define acc-in (acc-reshape input-shape (acc-list->tensor numbers)))
    (define-values (flat-res flat-cpu flat-real flat-gc) (time-apply (λ () (flat-sum-ρ flat-in)) '()))
    (define-values (acc-res acc-cpu acc-real acc-gc) (time-apply (λ () (acc-sum-ρ acc-in)) '()))
    #;(fprintf stdout "Flat Input: ~a~n" (flat-make-printable flat-in))
    #;(fprintf stdout "Flat Output: ~a~n" (flat-make-printable (list-ref flat-res 0)))
    #;(fprintf stdout "Accelerated Input: ~a~n" (acc-make-printable acc-in))
    #;(fprintf stdout "Accelerated Output: ~a~n" (acc-make-printable (list-ref acc-res 0)))
    (printf "~a, ~a, ~a, ~a, ~a, ~a, ~a~n"
            inner-size acc-cpu acc-real acc-gc flat-cpu flat-real flat-gc)))

(define (main)
  (define prim1-0-functions '("sqrt" "rectify" "abs" "exp" "log"))
  #;(for ((function prim1-0-functions))
    (call-with-output-file (format "benchmark-~a.csv" function) #:exists 'replace
    (λ (out)
      (parameterize ((current-output-port out))
        (fprintf stdout "Benchmarking ~a...~n" function)
        (benchmark-prim1 (string->procedure (format "flat-~a-ρ" function))
                         (string->procedure (format "acc-~a-ρ" function))
                         function)))))
  (define prim1-1-functions '(#;"sum" #;"max" "argmax"))
  #;(for ((function prim1-1-functions))
    (call-with-output-file (format "benchmark-~a.csv" function) #:exists 'replace
    (λ (out)
      (parameterize ((current-output-port out))
        (fprintf stdout "Benchmarking ~a...~n" function)
        (benchmark-prim1 (string->procedure (format "flat-~a-ρ" function))
                         (string->procedure (format "acc-~a-ρ" function))
                         function
                         '(100))))))
  (call-with-output-file "benchmark-sum-saturation.csv"  #:exists 'replace
    (λ (out)
      (parameterize ((current-output-port out))
        (fprintf stdout "Benchmarking sum saturation...~n")
        (benchmark-sum)))))
(main)
