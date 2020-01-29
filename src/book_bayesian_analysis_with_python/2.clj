(ns book-bayesian-analysis-with-python.2
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py. py.- py..]]
            [book-bayesian-analysis-with-python.utils :refer [with-show]]
            [oz.core :as oz]
            [tech.ml.dataset :as ds]))

(require-python '[builtins :as pyb]
                '[pymc3 :as pm :bind-ns]
                '[numpy :as np]
                '[arviz :as az]
                '[scipy.stats :as stats :refer [bernoulli]])


(pyb/help pm)
(assert (= 11 (np/dot [1 2 ] [ 3 4])))
(py.- pm "__version__")
(py. bernoulli rvs :p 0.4 :size 5)











(require-python '([scipy.stats :as stats :refer [bernoulli]]))
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


(az/summary trace)

(with-show (az/plot_trace trace))






(def data-chemical-shifts (np/loadtxt "./resources/data/chemical_shifts.csv"))

(let [ds (map #(zipmap [:x] %&) data-chemical-shifts)
      viz-spec {:data {:values ds}
                :mark :area
                :transform [{:density :x :bandwith 0.3}]
                :encoding {:x {:field :value :type :quantitative}
                           :y {:field :density :type :quantitative}}}
      ]
  (oz/view! viz-spec))

(def trace-g (py/with
            [model-g (py. pm Model)]
            (let [
                  mu (py. pm Uniform "mu" :lower 40 :upper 70)
                  sigma (py. pm HalfNormal "sigma" :sd 10)
                  y (py. pm Normal "y" :mu mu :sd sigma :observed data-chemical-shifts)
                  trace (py. pm sample 1000 :random_seed 123)]
              trace)))

(py/->jvm (az/summary trace-g))

(with-show (az/plot_trace trace-g))

(with-show (az/plot_joint trace-g :kind :kde :fill_last false))

(py/get-item trace-g :sigma)

;; (:sigma trace-g)  ;; doensn't work. 







(def inference-g (py/with
              [model-g (py. pm Model)]
              (let [
                    mu (py. pm Uniform "mu" :lower 40 :upper 70)
                    sigma (py. pm HalfNormal "sigma" :sd 10)
                    y (py. pm Normal "y" :mu mu :sd sigma :observed data-chemical-shifts)
                    trace (py. pm sample 1000 :random_seed 123)]
                {:trace  trace
                 :model model-g})))


(def y-pred-g
  (pm/sample_posterior_predictive (:trace inference-g) 100 (:model inference-g)))

(let [data-ppc (az/from_pymc3 :trace (:trace inference-g)
                              :posterior_predictive y-pred-g)]
  (with-show (az/plot_ppc data-ppc)))



(def inference-g (py/with
                  [model-g (py. pm Model)]
                  (let [
                        mu (py. pm Uniform "mu" :lower 40 :upper 70)
                        sigma (py. pm HalfNormal "sigma" :sd 10)
                        y (py. pm Normal "y" :mu mu :sd sigma :observed data-chemical-shifts)
                        trace (py. pm sample 1000 :random_seed 123)]
                    {:trace  trace
                     :model model-g})))


(py. (py. scipy.stats t :loc 0 :scale 1 :df 1) rvs 100)


(require-python '([scipy.stats :as stats :refer [t bernoulli]]))
(py. t             rvs :df 1 :size 3)
(py. stats/t       rvs :df 1 :size 3)
(py. scipy.stats/t rvs :df 1 :size 3)

(np/mean (py. t rvs :df 0.9 :size 100))



(def inference-gt (py/with
                  [model-gt (py. pm Model)]
                  (let [
                        mu (py. pm Uniform "mu" :lower 40 :upper 75)
                        sigma (py. pm HalfNormal "sigma" :sd 10)
                        nu (py. pm Exponential "nu" :lam 1/30 )
                        y (py. pm StudentT "y" :mu mu :sd sigma :nu nu
                               :observed data-chemical-shifts)
                        trace (py. pm sample 1000 :random_seed 123)]
                    {:trace  trace
                     :model model-gt})))

(with-show (az/plot_trace (:trace inference-gt)))

(let [{:keys [trace model]} inference-gt
      y-ppc-t (pm/sample_posterior_predictive trace 100 model :random_seed 123)
      y-pred-t (az/from_pymc3 :trace trace :posterior_predictive y-ppc-t)]
  (with-show
    (az/plot_ppc y-pred-t)
    (matplotlib.pyplot/xlim 40 70)))


(def tips (ds/->dataset "resources/data/tips.csv"))

(take 5 tips) ;; get first  5 columns

(ds/select tips :all (range 5)) ;; first 5 rows

(let [ds (ds/->flyweight tips)
      spec {:data  {:values ds}
            :mark :area
            :transform [{:density :tip :groupby [:day]}]
            :encoding {:x {:field :value :type :quantitative}
                       :y {:field :density :type :quantitative}
                       :color {:field :day :type :nominal}
                       :row {:field :day :type :nominal}}}]
  (oz/view! spec))

(require-python 'pandas)

(def tips-pd (pandas/read_csv "./resources/data/tips.csv"))

(def tip-pd (py/get-item tips-pd :tip))

(def idx-pd (py.- (pandas/Categorical (py/get-item tips-pd :day) :categories ["Thur" "Fri", "Sat" "Sun"]) codes))


(def inference-cg (py/with
                   [model (py. pm Model)]
                   (let [n-groups 4
                         mu (py. pm Normal "mu" :mu 0 :sd 10 :shape n-groups)
                         sigma (py. pm HalfNormal "sigma" :sd 10 :shape n-groups )
                         y (py. pm Normal "y"
                                :mu (py/get-item mu idx-pd)
                                :sd (py/get-item sigma idx-pd)
                                :observed tip-pd)
                         trace (py. pm sample 1000 :random_seed 123)]
                     {:trace  trace
                      :model model})))

(with-show(az/plot_trace (:trace inference-cg)))


;; and now without pandas

(def tips (ds/->dataset "./resources/data/tips.csv"))
(def tip (ds/column tips "tip"))

;; (require '[tech.ml.dataset.categorical])
(require '[tech.ml.dataset.pipeline :as ds-pipe])

(def tips-numberfied (ds-pipe/string->number tips "day"))

(ds/unique-by "day" tips-numberfied)
(tech.ml.dataset.column/unique (tips "day"))
(tech.ml.dataset.categorical/build-categorical-map tips (list "day"))

(def idx (tech.ml.dataset.categorical/column-categorical-map
          (tech.ml.dataset.categorical/build-categorical-map tips (list "day"))
          :int32
          (ds/column tips "day")))    ;; TODO this has got to have easier method

;; (defn col-dtype->int [ds column-name]
;;   (tech.ml.dataset.categorical/column-categorical-map
;;    (tech.ml.dataset.categorical/build-categorical-map ds (list column-name))
;;    :int32
;;    (tech.ml.dataset/column ds column-name)))

;; (col-dtype->int tips "day")

(def tips-int(-> (tech.ml.dataset.pipeline/string->number tips)
                 (tech.ml.dataset.pipeline/->datatype  tech.ml.dataset.pipeline.column-filters/categorical? :int32)))


(def idx2 (tech.ml.dataset.categorical/column-categorical-map
          (tech.ml.dataset.categorical/build-categorical-map tips (list "day"))
          :int32
          (ds/column tips "day")))    ;; TODO this has got to have easier method

(tech.ml.dataset.categorical/column-values->categorical tips
                                                        "day"
                                                        (tech.ml.dataset.categorical/build-categorical-map tips (list "day"))

                                                        )
(tech.ml.dataset.pipeline/string->number tips)
(tech.ml.dataset.pipeline/->datatype (tech.ml.dataset.pipeline/string->number tips) (tech.ml.dataset.pipeline.column-filters/select-columns ["day"] tips) :int32)

(tech.ml.dataset.pipeline/->datatype 
 tips
 ;; (tech.ml.dataset.pipeline.column-filtrs/select-columns ["day"] tips)


 (tech.ml.dataset.pipeline.column-filters/select-column-names (list "day") tips)
 :int32)

(ds/select-columns tips nil)

 ;; (tech.ml.dataset.pipeline/string->number tips) (tech.ml.dataset.pipeline.column-filters/select-columns ["day"] tips) :int32)


(require 'tech.libs.tablesaw.tablesaw-column)
(tech.libs.tablesaw.tablesaw-column/datatype->column-data-cast-fn :int32 (tips-int "day"))

(tech.ml.dataset.column/new-column "blah" :int32 (tips-int "day"))

(require '[tech.v2.datatype :as dtype])

(ds/update-column tips-int "day" #(dtype/->reader % :float64))

(dtype/->float-array (tips-int "day"))

(def tips-numbered (tech.ml.dataset.pipeline/string->number tips))

(dtype/->int-array (tips-numbered "day"))

(take 20 (tips "day"))

(string)

;;;;; here

;; Add, Remove, Update

;; Adding or updating columns requires either a fully constructed column (dtype/make-container :tablesaw-column :float32 elem-seq) or a reader that has a type compatible with tablesaw's column system. For this reason you may be errors if you pass a persistent vector in to the add-or-update method without first given it a datatype via (dtype/->reader [1 2 3 4] :float32).

;; user> (require '[tech.v2.datatype.functional :as dfn])
;; nil
;; ;;Log doesn't work if the incoming value isn't a float32 or a float64.  SalePrice is
;; ;;of datatype :int32 so we convert it before going into log.
;; user> (ds/update-column small-ames "SalePrice" #(-> (dtype/->reader % :float64)
;;                                                     dfn/log))
;; [5 2]:


;; '' tile here 

(def )
(def n-groups (count (set idx)))

(def tip (ds/column tips "tip"))

(set idx)

(def idx-ds
  (->(tech.ml.dataset.pipeline/string->number tips)
     (ds/column "day")
     (dtype/->reader :int32)
     ))

(def inference-cg2 (py/with
                   [model (py. pm Model)]
                   (let [mu (py. pm Normal "mu" :mu 0 :sd 10 :shape n-groups)
                         sigma (py. pm HalfNormal "sigma" :sd 10 :shape n-groups )
                         y (py. pm Normal "y"
                                :mu (py/get-item mu idx-ds)
                                :sd (py/get-item sigma idx-ds)
                                :observed (tips "tip"))
                         trace (py. pm sample 1000 :random_seed 123)]
                     {:trace  trace
                      :model model})))

(with-show(az/plot_trace (:trace inference-cg2)))

(py/dir(:trace inference-cg2))
(pyb/help(:trace inference-cg2))

(-> (:trace inference-cg2)
    (py/get-item "mu")
    ;; (py/get-item 2)
    tech.v2.tensor/->tensor
    )
