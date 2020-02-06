(ns book-bayesian-analysis-with-python.2
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py. py.- py..]]
            [book-bayesian-analysis-with-python.utils :refer [with-show]]
            [oz.core :as oz]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.pipeline :as ds-pipe]
                     ))

(require-python '[builtins :as python]
                '[pymc3 :as pm :bind-ns]
                '[numpy :as np]
                '[arviz :as az]
                '[scipy.stats :as stats :refer [bernoulli beta]]
                'operator)


(python/help pm)
(assert (= 11 (np/dot [1 2 ] [ 3 4])))
(py.- pm "__version__")
(py. bernoulli rvs :p 0.4 :size 5)










;; (require-python '([scipy.stats :as stats :refer [bernoulli]]))

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

(def tips-raw (ds/->dataset "./resources/data/tips.csv"))

(def tips-ds
  (-> tips-raw
      (ds-pipe/string->number "day" )
      (ds-pipe/->datatype "day" :int32)))


(def inference-cg2
  (let [n-groups (count (set (tips-ds "day")))
        tip (tips-ds "tip")
        idx (tips-ds "day")]
    (py/with
     [model (py. pm Model)]
     (let [mu    (pm/Normal "mu"
                            :mu 0
                            :sd 10
                            :shape n-groups)
           sigma (pm/HalfNormal "sigma"
                                :sd 10
                                :shape n-groups)
           y     (pm/Normal "y"
                            :mu (py/get-item mu idx)
                            :sd (py/get-item sigma idx)
                            :observed tip)
           diffs (doseq [i (range n-groups)
                         j (range n-groups)
                         :when (< i j)]
                   (pm/Deterministic
                    (str "mu" i " - mu" j)
                    (operator/sub (py/get-item mu i)
                                  (py/get-item mu j))))
           trace (pm/sample 1000 :random_seed 123)]
       {:trace  trace
        :model model}))))


(with-show
  (az/plot_forest (:trace inference-cg2)))


(def n-samples (repeat 3 30))
(def g-samples (repeat 3 18))

(def group-idx (repeat  (count n-samples)))


(defn sim-data-h
  [n-samples g-samples]
  (vec (mapcat (fn [g n] (concat (repeat g 1) (repeat (- n g) 0))) g-samples n-samples)))
(sim-data-h n-samples g-samples)

(defn build-group-idx-h [n-samples]
  (vec (mapcat (fn [i el] (repeat el i)) (range (count n-samples)) n-samples)))

#_(def inference-h
  (let [{:keys [group-idx data]} (sim-data-h n-samples g-samples)
        n-groups (count (set group-idx))]
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Beta "mu" 1.0 1.0)
           kappa (pm/HalfNormal "kappa" 10)
           theta (pm/Beta "theta"
                          :alpha (operator/mul mu kappa)
                          :beta (operator/mul kappa (operator/sub 1.0 mu))
                          :shape n-groups)
           y     (pm/Bernoulli "y"
                               :p (py/get-item theta group-idx)
                               :observed data)
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model}))))

(with-show (az/plot_trace (:trace inference-h)))

(az/summary (:trace inference-h))

(defn run-inference-h
  [n-samples g-samples]
  (let [group-idx (build-group-idx-h n-samples)
       data (sim-data-h n-samples g-samples)]
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Beta "mu" 1.0 1.0)
           kappa (pm/HalfNormal "kappa" 10)
           theta (pm/Beta "theta"
                          :alpha (operator/mul mu kappa)
                          :beta (operator/mul kappa (operator/sub 1.0 mu))
                          :shape (count (set group-idx)))
           y     (pm/Bernoulli "y"
                               :p (py/get-item theta group-idx)
                               :observed data)
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model}))))

(def inference-h
  (run-inference-h n-samples g-samples))

(let [g-samples [18 3 3]
      {:keys [trace]} (run-inference-h n-samples g-samples)]
  (with-show (az/plot_trace trace))
  (az/summary trace))

(with-show (az/plot_joint (:trace inference-h) ))

(with-show (az/plot_ppc (az/from)(:trace inference-h)
                        :var_names ["theta"]
                        ))

(def y-ppc-t (pm/sample_posterior_predictive (:trace inference-h)
                                              100
                                              (:model inference-h)
                                              :random_seed 123))
(def y-pred-t (az/from_pymc3 :trace (:trace inference-h)
                             :posterior_predictive y-ppc-t))
#_((with-show
     (az/plot_ppc y-pred-t :var_names ["kappa"])))

(py/get-item (:trace inference-h) "theta")

(with-show (az/plot_ppc (:trace inference-h) :var_names ["kappa"]))

;; TODO don't know how to make this "theta" prior yet.
;; can I use the theta vars for this?
;; or should I create new Deterministic var based on kappa and mu?


(defn run-inference-h2
  [n-samples g-samples]
  (let [group-idx (build-group-idx-h n-samples)
        data (sim-data-h n-samples g-samples)]
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Beta "mu" 1.0 1.0)
           kappa (pm/HalfNormal "kappa" 10)
           theta (pm/Beta "theta"
                          :alpha (operator/mul mu kappa)
                          :beta (operator/mul kappa (operator/sub 1.0 mu))
                          :shape (count (set group-idx)))
           y     (pm/Bernoulli "y"
                               :p (py/get-item theta group-idx)
                               :observed data)
           prior_theta
           (pm/Beta "prior_theta"
                    :alpha (operator/mul mu kappa)
                    :beta (operator/mul kappa (operator/sub 1.0 mu)))
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model}))))

(def inference-h2 (run-inference-h2 n-samples g-samples))

(def y-ppc-t-h2 (pm/sample_posterior_predictive (:trace inference-h2)
                                             50
                                             (:model inference-h2)
                                             :random_seed 123
                                             :var_names ["mu" "kappa" "prior_theta" ]
                                            ))


(def y-pred-t-h2 (az/from_pymc3 :trace (:trace inference-h2)
                             :posterior_predictive y-ppc-t-h2))

(py/get-item y-pred-t-h2 "posterior_predictive")


(with-show (az/plot_ppc y-pred-t-h2 :var_names ["kappa"]
                        ))


(map (fn [mu kappa] (py. (stats/beta (* mu kappa)
                                     (* kappa (- 1 mu)))
                         pdf (np/linspace 0 1 100)))
     (y-ppc-t-h2 "mu") (y-ppc-t-h2 "kappa"))


(def prior-theta-lines (mapcat (fn [mu kappa]
                             (for [x (np/linspace 0 1 40)]
                               {:group (str "kappa: " kappa ", mu " mu)
                                        ;:kappa kappa
                                        ;:mu mu
                                :x x
                                :pdf (py. (stats/beta (* mu kappa)
                                                      (* kappa (- 1 mu)))
                                          pdf x)}))
                           (y-ppc-t-h2 "mu") (y-ppc-t-h2 "kappa")))

(take 5 prior-theta-lines)

(oz/view!
 {:data {:values prior-theta-lines}
  :mark :line
  :encoding {:x {:field :x :type :quantitative}
             :y {:field :pdf :type :quantitative}
             :detail {:field :group :type :nominal}
             :opacity {:value 0.5}
             :size {:value 0.6}}})

(def prior-theta-lines2
  (mapcat (fn [mu kappa]
            (for [x (np/linspace 0 1 40)]
              {:group (str "kappa: " kappa ", mu " mu)
                                        ;:kappa kappa
                                        ;:mu mu
               :x x
               :pdf (py. (stats/beta (* mu kappa)
                                     (* kappa (- 1 mu)))
                         pdf x)}))
          (take 40 (py/get-item (:trace inference-h2) "mu"))
          (take 40 (py/get-item (:trace inference-h2) "kappa"))))

(py/get-item (:trace inference-h2) "mu") 

(take 5 prior-theta-lines2)
(count prior-theta-lines2)

(oz/view!
 {:data {:values prior-theta-lines2}
  :mark :line
  :encoding {:x {:field :x :type :quantitative}
             :y {:field :pdf :type :quantitative}
             :detail {:field :group :type :nominal}
             :opacity {:value 0.5}
             :size {:value 0.6}}})

(with-show (az/plot_pair (:trace inference-h2) :kind :kde :var_names ["mu" "kappa"]))




;; one more example

(def chem-shifts-theo-exp
  (ds/->dataset "./resources/data/chemical_shifts_theo_exp.csv"))


(def chem-shifts-theo-exp
  (-> (ds/->dataset "./resources/data/chemical_shifts_theo_exp.csv")
      (ds-pipe/string->number "aa")
      (ds-pipe/->datatype "aa" :int32)))

(require '[tech.v2.datatype.functional :as dfn])
(def diff (dfn/- (chem-shifts-theo-exp "theo") (chem-shifts-theo-exp "exp")))



(def aa->indices
  (->> chem-shifts-theo-exp
       ds/->flyweight
       (map #(get % "aa"))
       set
       (into {} (map-indexed (fn [i e] [e i])) )
       ))

(defn add-diff [df]
  (map (fn [{:strs [theo exp] :as m}]
         (assoc m :diff (- theo exp))) df))

(defn pipeline-nh [df ]
  (->> df
       ds/->flyweight
       add-diff
       (map #(update % "aa" aa->indices))
       ))

(pipeline-nh chem-shifts-theo-exp)


;; TODO see if this is easier to do with meander. I don't think
;; dataset is a good solution yet. Doesn't feel like finished thing.

(def groups-nh (count aa->indices))

(def idx-nh
  (->> (pipeline-nh chem-shifts-theo-exp)
       (mapv #(get % "aa"))))

(def data-nh
  (->> (pipeline-nh chem-shifts-theo-exp)
       (mapv :diff)))



(def run-inference-nh
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Normal "mu" :mu 0 :sd 10 :shape groups-nh)
           sigma (pm/HalfNormal "sigma" :sd 10 :shape groups-nh)
           y     (pm/Normal "y"
                            :mu (py/get-item mu idx-nh)
                            :sd (py/get-item sigma idx-nh)
                            :observed data-nh)
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model})))



(def run-inference-h
  (py/with
   [model (pm/Model)]
   (let [
         mu-mu (pm/Normal "mu-mu" :mu 0 :sd 10)
         sigma-mu (pm/HalfNormal "sigma-mu" :sd 10)

         mu    (pm/Normal "mu" :mu mu-mu :sd sigma-mu :shape groups-nh)
         sigma (pm/HalfNormal "sigma" :sd 10 :shape groups-nh)
         y     (pm/Normal "y"
                          :mu (py/get-item mu idx-nh)
                          :sd (py/get-item sigma idx-nh)
                          :observed data-nh)
         trace (pm/sample 2000 :random_seed 123)]
     {:trace  trace
      :model model})))

;; TODO find some way to build a macro/function that takes the name of a symbol and adds
;; it as the first parameter instead of me having to type the symbol AND the string.


(with-show
  (az/plot_forest (mapv :trace [run-inference-h run-inference-nh])
                  :var_names "mu"
                  :combined false
                  :colors "cycle"))


(def run-inference-h-swapped-order ; gives error
  (py/with
   [model (pm/Model)]
   (let [
         mu    (pm/Normal "mu" :mu mu-mu :sd sigma-mu :shape groups-nh)

         mu-mu (pm/Normal "mu-mu" :mu 0 :sd 10)
         sigma-mu (pm/HalfNormal "sigma-mu" :sd 10)

         sigma (pm/HalfNormal "sigma" :sd 10 :shape groups-nh)
         y     (pm/Normal "y"
                          :mu (py/get-item mu idx-nh)
                          :sd (py/get-item sigma idx-nh)
                          :observed data-nh)
         trace (pm/sample 2000 :random_seed 123)]
     {:trace  trace
      :model model})))

;; TODO create something in which I give the model, where the order doesn't matter.
;; so some map like data structure. either some map with maybe some plumbing liek features, or maybe some hiccup syntax or even some datalog syntax
 
