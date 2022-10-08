#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/generic)

;; ----------------------------------------------------

(require flomat)
(require plot)

;; ----------------------------------------------------

(require "activation.rkt")
(require "dnn.rkt")
(require "ga.rkt")
(require "layer.rkt")
(require "loss.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define dqn%
  (class dnn%
    (init model)

    ; constructor fields
    (init-field initial-state
                state->X
                perform-action

                ; learning fields
                [gamma 0.9]
                [batch-size 500])

    ; batch count for this generation
    (define batch 0)

    ; define a custom agent class
    (define agent%
      (class* object% (recomb<%>)
        (super-new)

        ; constructor fields
        (init-field model)

        ; state fields
        (field [state (initial-state)]
               [reward 0]
               [terminal? #f])
        
        ; revert the state of the agent
        (define/public (reset-state)
          (set! state (initial-state))
          (set! reward 0)
          (set! terminal? #f))

        ; run input vector
        (define/public (step)
          (or terminal?

              ; choose action, perform, and update state
              (let* ([X (state->X state)]
                     [Z (send model call X)])
                (let-values ([(new-reward new-state new-terminal?)
                              (perform-action state (.argmax Z))])
                  (begin0 new-terminal?

                          ; update state fields
                          (set! state new-state)
                          (set! terminal? new-terminal?)
                          (set! reward (+ reward new-reward)))))))
        
        ; recombine with another agent
        (define/public (recomb other)
          (new this% [model (send model recomb (get-field model other))]))))

    ; initialize the population with agent models
    (super-new [model (λ () (new agent% [model (model)]))])

    ; allow use of the get-model method
    (inherit [get-agent get-model])
    
    ; allow use of all the models
    (inherit-field [agents models])

    ; return the state of the best agent
    (define/public (get-state)
      (get-field state (get-agent)))

    ; have the best agent step and perform an action
    (define/public (step)
      (send (get-agent) step))

    ; advance to the next generation
    (define/private (next-gen)
      (begin0 (next-gen! agents (λ (agent) (get-field reward agent)) >)

              ; reset batch counter
              (set! batch 0)
      
              ; reset agents
              (for ([agent agents])
                (send agent reset-state))))
    
    ; perform actions for each agent
    (define/override (train #:watch? [watch? #f])
      (set! batch (add1 batch))

      ; true if all agents are terminal or batch is full
      (and (or (for/fold ([all-terminal? #t])
                         ([agent agents])
                 (let ([terminal? (send agent step)])
                   (and all-terminal? terminal?)))
               
               ; batch full?
               (and batch-size (>= batch batch-size))
               
               ; watched agent in terminal state
               (and watch? (get-field terminal? (get-agent))))

           ; advance agents to the next generation
           (next-gen)))

    ; train agents for multiple generations
    (define/override (train+ #:generations [n 100])
      (let ([reward (for/list ([x n])
                      (do ([y (train)
                              (train)])
                        [y (list x y)]))])
        (plot (lines reward #:y-min 0) #:x-label "Generation" #:y-label "Reward")))))
