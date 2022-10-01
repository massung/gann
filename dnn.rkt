#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)
(require plot)

;; ----------------------------------------------------

(require "activation.rkt")
(require "ga.rkt")
(require "layer.rkt")
(require "loss.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define dnn%
  (class* object% ()
    (super-new)

    ; constructor fields
    (init-field model [population-size 100])

    ; create the initial population of models
    (field [models (build-vector population-size (λ (_) (model)))]

           ; current generation
           [generation 0])

    ; get a model
    (define/public (get-model [i 0])
      (vector-ref models i))

    ; use the best model to predict
    (define/public (predict X)
      (send (get-model) predict X))
    
    ; determine the fitness for each model in the population
    (define (fit fitness #:less-than? [less-than? <])
      (let ([index (for/vector ([(model i) (in-indexed models)])
                     (cons i (fitness model)))])
        
        ; sort the index by fitness
        (vector-sort! index less-than? #:key cdr)

        ; return the best fitness and elite models (top 10%)
        (values (cdr (vector-ref index 0))

                (let ([n (quotient population-size 10)])
                  (for/vector ([_ n] [i index])
                    (vector-ref models (car i)))))))

    ; crossover models to make a new population
    (define/public (next-gen fitness #:less-than? [less-than? <])
      (let-values ([(loss elite) (fit fitness #:less-than? less-than?)])

        ; save the best models for the next generation
        (vector-copy! models 0 elite)

        ; increase the generation count
        (set! generation (add1 generation))

        ; return the best fitness
        (begin0 loss
                  
                ; select random parent from elite models
                (let ([parent (λ () (random (vector-length elite)))])

                  ; generate new children for the rest of the generation
                  (for ([i (in-range (vector-length elite) population-size)])
                    (let ([a (vector-ref elite (parent))]
                          [b (vector-ref elite (parent))])
                      (vector-set! models i (send a crossover b))))))))

    ; train multiple generations
    (define/public (train Xs Ys [loss mean-squared] #:generations [n 100])
      (let* ([fitness (λ (model)
                        (for/sum ([X Xs] [Y (map matrix Ys)])
                          (let ([Z (send model predict X)])
                            (loss Y Z))))]

             ; produce generations using the fitness
             [next (in-producer (λ () (next-gen fitness)))]

             ; graph the loss over generations
             [xy (for/list ([x n] [y next])
                   (list x y))])
        (plot (lines xy #:y-min 0) #:x-label "Generation" #:y-label "Loss")))))

;; ----------------------------------------------------

(module+ test
  (define-model xor-model seq-model%
    [(dense-layer 5 .relu!)
     (dense-layer 1 .gaussian!)]
    #:inputs 2)

  ; create the xor neural network
  (define dnn (new dnn% [model xor-model]))

  ; construct training data set (X=input, Y=output)
  (define Xs '([0 0] [0 1] [1 0] [1 1]))
  (define Ys '([0]   [1]   [1]   [0]))

  ; train the network
  (time (send dnn train Xs Ys))

  ; run the test data to see the outputs
  (for ([X Xs])
    (let ([Z (send dnn predict X)])
      (displayln (format "~a -> ~a" X Z)))))
