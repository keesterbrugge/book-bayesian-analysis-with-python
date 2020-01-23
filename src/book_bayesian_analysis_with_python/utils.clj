(ns book-bayesian-analysis-with-python.utils
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py.]]))

;; try out plot functionality from gigasquid blo
;; https://gigasquidsoftware.com/blog/2020/01/18/parens-for-pyplot/

(require '[clojure.java.shell :as sh])

;;; This uses the headless version of matplotlib to generate a graph then copy it to the JVM
;; where we can then print it

;;;; have to set the headless mode before requiring pyplot
(def mplt (py/import-module "matplotlib"))
(py. mplt "use" "Agg")

(require-python 'matplotlib.pyplot)
(require-python 'matplotlib.backends.backend_agg)


(defmacro with-show
  "Takes forms with mathplotlib.pyplot to then show locally"
  [& body]
  `(let [_# (matplotlib.pyplot/clf)
         fig# (matplotlib.pyplot/figure)
         agg-canvas# (matplotlib.backends.backend_agg/FigureCanvasAgg fig#)]
     ~(cons 'do body)
     (py. agg-canvas# "draw")
     (matplotlib.pyplot/savefig "temp.png")
     (sh/sh "open" "temp.png")))

(comment

  (py/import-as numpy np)
  (let [x (np/arange 0 (* 3 np/pi) 0.1)
        y (np/sin x)]
    (with-show
      (matplotlib.pyplot/plot x y))) ;; NOTE works :)



  )
