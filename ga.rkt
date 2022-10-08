#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/generic)

;; ----------------------------------------------------

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define mutation-rate (make-parameter 0.03))
(define inversion-rate (make-parameter 0.50))

;; ----------------------------------------------------

(define (mutate x)
  (if (> (random) (mutation-rate))
      x
      (- (random) (random))))

;; ----------------------------------------------------

(define-pointwise-unary mutate)

;; ----------------------------------------------------

(define (crossover hap A B)
  (apply stack (for/list ([(i x) (in-col hap 0)])
                 (row (if (< x 0.5) A B) i))))

;; ----------------------------------------------------

(define recomb<%>
  (interface () recomb))

;; ----------------------------------------------------

(define (next-gen! pop fitness [less-than? <] #:elite-selection [n (quotient (vector-length pop) 10)])
  (let ([xs (vector-map (λ (x) (cons x (fitness x))) pop)])
    (vector-sort! xs less-than? #:key cdr)

    ; update the population, keep the best models
    (for ([(x i) (in-indexed xs)])
      (if (< i n)
          (vector-set! pop i (car x))
          
          ; create new models
          (let ([a (car (vector-ref xs (random n)))]
                [b (car (vector-ref xs (random n)))])
            (vector-set! pop i (send a recomb b)))))

    ; return the best fitness
    (cdr (vector-ref xs 0))))

;; ----------------------------------------------------

(module+ test
  (require rackunit)

  ; validate crossover code
  (let* ([A (flomat: [[1 1 1] [2 2 2] [3 3 3] [4 4 4]])]
         [B (flomat: [[5 5 5] [6 6 6] [7 7 7] [8 8 8]])]

         ; select haplotypes and crossover
         [C (crossover (column 0.1 0.2 0.7 0.8) A B)])
    (check-equal? C (flomat: [[1 1 1] [2 2 2] [7 7 7] [8 8 8]]))))
