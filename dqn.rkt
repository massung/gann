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
                perform-action)

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

    ; create a constructor for an agent model
    (define (agent)
      (new agent% [model (model)]))

    ; build the initial population
    (super-new [model agent])

    ; models are actually agents that have models in them
    (inherit-field [agents models])

    ; return a model from the agent population
    (define/override (get-model [i 0])
      (get-field model (get-agent i)))

    ; return an agent from the population
    (define/public (get-agent [i 0])
      (vector-ref agents i))

    ; return the state of an agent
    (define/public (get-state [i 0])
      (get-field state (get-agent i)))

    ; are all agents in a terminal state
    (define (all-terminal?)
      (for/and ([agent agents])
        (get-field terminal? agent)))
    
    ; perform actions for each agent
    (define/public (step #:train? [train? #t])
      (begin0 (for/list ([agent agents] #:unless (get-field terminal? agent))
                (send agent step)

                ; return the state for all agents
                (get-field state agent))

              ; if all agents are terminal, train the next generation
              (when (and train? (all-terminal?))
                (train))))

    ; train the next generation and return the best fitness
    (define/override (train)
      (begin0 (next-gen! agents (λ (agent) (get-field reward agent)) >)
      
              ; reset agents
              (for ([agent agents])
                (send agent reset-state))))

    ; train agents for multiple generations
    (define/override (train+ #:generations [n 100] #:steps [batch-size 100])
      (let ([reward (for/list ([x n])
                      (for ([_ batch-size] #:break (all-terminal?))
                        (step))
                      (list x (train)))])
        (plot (lines reward #:y-min 0) #:x-label "Generation" #:y-label "Reward")))))
