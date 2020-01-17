(ns book-bayesian-analysis-with-python.1
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]
            [oz.core :as oz]))

;; (py/initialize!)
;; (py/import-as pymc3 pm)
(require-python '([pymc3 :as pm]))
(require-python '([builtins :as pyb]))

(py/$. pm "__version__")
;; => "3.8"

(pyb/help pm)

;; first example and ways of runnig it
(-> (py/run-simple-string "import scipy; x = scipy.stats.norm(0,1).rvs(3)")
    :globals
    (get "x"))

(require-python 'scipy.stats)
(py/$a (scipy.stats/norm 0 1) rvs 3)

(require-python '[numpy :as np])

(defn fig1-1-spec [ds]
  {:data {:values ds}
   :mark :line
   :encoding {:x {:field :x}
              :y {:field :pdf}
              :row {:field :mu}
              :column {:field :sd}}
   :resolve {:axis {:x :independent :y :independent}}})

(def fig1-1-ds
  (apply concat (for [mu [-1 0 1]
                sd [0.5 1 1.5]
                ]
            (let [x (np/linspace -7 7 200)
                  pdf (-> (scipy.stats/norm :loc mu :scale sd)
                          (py/$a pdf x ))]
              (map #(hash-map :x %1 :pdf %2 :mu %3 :sd sd) x pdf (repeat mu))
              )))
  )

(oz/view! (fig1-1-spec fig1-1-ds))




