(ns book-bayesian-analysis-with-python.2
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py. py.- py..]]
            [oz.core :as oz]))

;; (py/finalize!)
;; (py/initialize!)
;; (py/import-as pymc3 pm)
;; (py/import-as numpy np)


(require-python '([builtins :as pyb]))
(require-python '([pymc3 :as pm]))
(pyb/help pm)
(require-python '([numpy :as np]))
(np/dot [1 2 ] [ 3 4])

(py.- pm "__version__")
;; => "3.8"



(require-python '([scipy.stats :refer [bernoulli]]))

(py. np/random seed 123)
(def trials 4)
(def theta-real 0.35)
(def data (py. bernoulli rvs :p theta-real :size trials))

(def trace (py/with
            [our-first-model (py. pm Model)]
            (let [theta (py. pm Beta "theta" :alpha 1 :beta 1)
                  y (py. pm Bernoulli "y" :p theta :observed data)
                  trace (py. pm sample 1000 :random_seed 123)]
              trace)))


(require-python '([arviz :as az]))
(az/summary trace)

(with-show (az/plot_trace trace))

















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

(let [x (np/arange 0 (* 3 np/pi) 0.1)
      y (np/sin x)]
  (with-show
    (matplotlib.pyplot/plot x y))) ;; NOTE works :)


(with-show (az/plot_trace trace))


(defn fig1-1-spec [ds]
  {:data {:values ds}
   :mark :line
   :encoding {:x {:field :x}
              :y {:field :pdf}
              :row {:field :mu}
              :column {:field :sd}}
   :resolve {:axis {:x :independent :y :independent}}})

(def fig1-1-ds
  (mapcat seq (for [mu [-1 0 1]
                sd [0.5 1 1.5]
                ]
            (let [x (np/linspace -7 7 200)
                  pdf (-> (scipy.stats/norm :loc mu :scale sd)
                          (py/$a pdf x ))]
              (map #(hash-map :x %1 :pdf %2 :mu mu :sd sd) x pdf)
              ))))

(oz/view! (fig1-1-spec fig1-1-ds))



(def fig1-3-ds
  (let [n-params [1 2 4]
        p-params [0.25 0.5 0.75]
        xs (np/arange 0 (inc (np/max n-params)))]
    (for [n n-params p p-params x xs]
      ;; {:pmf (py/$a (scipy.stats/binom n p) pmf x)
      {:pmf (py. (scipy.stats/binom n p) pmf x)
       :x x
       :n n
       :p p})))

(defn fig1-3-spec [ds]
  {:data {:values ds}
   :encoding {:x {:field :x}
              :y {:field :pmf}
              :row {:field :n}
              :column {:field :p}}
   :mark :bar
   :resolve {:axis {:x :independent :y :independent}}})

(oz/view! (fig1-3-spec fig1-3-ds))

(require 'clojure.java.io)

(clojure.java.io/file "/Users/keesterbrugge/Downloads/clojure.png" )
