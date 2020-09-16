(defproject book-bayesian-analysis-with-python "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  #_:dependencies #_[[org.clojure/clojure "1.10.0"]
                 [clj-python/libpython-clj "1.37"]
                 [metasoarous/oz "1.6.0-alpha6"]
                 #_[techascent/tech.ml.dataset "b8a8ba79d1b45879da5d723edc08be15f6ea0b1f"]
                 [techascent/tech.ml.dataset "2.0-alpha-2-SNAPSHOT"]
                 #_[techascent/tech.ml.dataset "1.73"]
                 [meander/epsilon "0.0.402"]]
  ;; :plugins [[reifyhealth/lein-git-down "0.3.5"]]
  ;; :repositories [["public-github" {:url "git://github.com"}]]
  :plugins [[lein-tools-deps "0.4.5"]]
  :middleware [lein-tools-deps.plugin/resolve-dependencies-with-deps-edn]
  :lein-tools-deps/config {:config-files [:install :user :project]}
  :repl-options {:init-ns book-bayesian-analysis-with-python.core})
