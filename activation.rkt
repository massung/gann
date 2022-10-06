#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define (gaussian x) (exp (- (* x x))))
(define (relu x) (max 0.0 x))
(define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))
(define (softplus x) (log (+ 1.0 (exp x))))
(define (step x) (if (< x 0.0) 0.0 1.0))

;; ----------------------------------------------------

(define-pointwise-unary gaussian)
(define-pointwise-unary identity)
(define-pointwise-unary relu)
(define-pointwise-unary sigmoid)
(define-pointwise-unary softplus)
(define-pointwise-unary step)
(define-pointwise-unary tanh)

;; ----------------------------------------------------

(define (.argmax Z)
  (let ([q (for/fold ([q #f]) ([(i z) (in-col Z 0)])
             (if (zero? i) z (max q z)))])
    (apply column (for/list ([(_ z) (in-col Z 0)])
                    (if (= z q) 1.0 0.0)))))

;; ----------------------------------------------------

(define (.softmax Z)
  (let ([e (.exp Z)])
    (./! e (for/sum ([(_ j) (in-col e 0)]) j))))

;; ----------------------------------------------------

(define (batch-mean Z)
  (/ (for/sum ([(_ j) (in-col Z 0)]) j) (size Z)))

;; ----------------------------------------------------

(define (batch-normalized activation [epsilon 0.001])
  (λ (Z)
    (let* ([n (batch-mean Z)]
         
           ; squared dev. from mean
           [m (/ (for/sum ([(_ j) (in-col Z 0)])
                   (let ([d (- j n)])
                     (* d d)))
                 (size Z))])
      
      ; normalized Z
      (activation (if (< m epsilon)
                      Z
                      (./! (.- Z n) (sqrt (- m epsilon))))))))
