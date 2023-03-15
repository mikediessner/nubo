Search.setIndex({"docnames": ["asynchronous_bo", "bayesian_gp", "bayesian_optimisation", "citation", "constrained_bo", "get_started", "index", "multipoint_joint", "multipoint_sequential", "nubo.acquisition", "nubo.models", "nubo.optimisation", "nubo.test_functions", "nubo.utils", "overview", "singlepoint"], "filenames": ["asynchronous_bo.ipynb", "bayesian_gp.ipynb", "bayesian_optimisation.rst", "citation.md", "constrained_bo.ipynb", "get_started.ipynb", "index.rst", "multipoint_joint.ipynb", "multipoint_sequential.ipynb", "nubo.acquisition.rst", "nubo.models.rst", "nubo.optimisation.rst", "nubo.test_functions.rst", "nubo.utils.rst", "overview.rst", "singlepoint.ipynb"], "titles": ["Asynchronous Bayesian optimisation", "Fully Bayesian Gaussian process for Bayesian optimisation", "A primer on Bayesian optimisation", "Citation", "Constrained Bayesian optimisation", "Get started", "NUBO: a transparent python package for Bayesian optimisation", "Parallel multi-point joint Bayesian optimisation", "Parallel multi-point sequential Bayesian optimisation", "Acquisition module", "Surrogates module", "Optimisation module", "Test function module", "Utility module", "Overview", "Sequential single-point Bayesian optimisation"], "terms": {"1": [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], "import": [0, 1, 4, 5, 7, 8, 15], "torch": [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], "from": [0, 1, 2, 4, 5, 6, 7, 8, 11, 13, 14, 15], "nubo": [0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15], "acquisit": [0, 1, 4, 5, 6, 7, 8, 11, 14, 15], "mcexpectedimprov": [0, 7, 8, 9], "mcupperconfidencebound": [0, 7, 8, 9], "model": [0, 1, 4, 5, 6, 7, 8, 9, 10, 14, 15], "gaussianprocess": [0, 1, 4, 5, 7, 8, 10, 15], "fit_gp": [0, 4, 5, 7, 8, 10, 15], "joint": [0, 2, 6, 11], "test_funct": [0, 1, 4, 5, 7, 8, 12, 15], "hartmann6d": [0, 1, 4, 5, 7, 8, 12, 15], "util": [0, 1, 4, 5, 6, 7, 8, 15], "gen_input": [0, 1, 4, 5, 7, 8, 13, 15], "gpytorch": [0, 1, 2, 4, 5, 7, 8, 9, 10, 14, 15], "likelihood": [0, 1, 2, 4, 5, 7, 8, 10, 14, 15], "gaussianlikelihood": [0, 1, 4, 5, 7, 8, 15], "test": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15], "function": [0, 1, 3, 4, 6, 7, 8, 10, 11, 14, 15], "func": [0, 1, 4, 5, 7, 8, 11, 15], "minimis": [0, 1, 4, 5, 7, 8, 11, 12, 15], "fals": [0, 1, 4, 5, 7, 8, 9, 12, 15], "dim": [0, 1, 4, 5, 7, 8, 9, 12, 13, 15], "bound": [0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15], "train": [0, 1, 2, 4, 5, 7, 8, 9, 10, 14, 15], "data": [0, 1, 2, 4, 5, 7, 8, 9, 14, 15], "x_train": [0, 1, 4, 5, 7, 8, 10, 15], "num_point": [0, 1, 4, 5, 7, 8, 13, 15], "5": [0, 1, 2, 4, 5, 7, 8, 10, 14, 15], "num_dim": [0, 1, 4, 5, 7, 8, 13, 15], "y_train": [0, 1, 4, 5, 7, 8, 10, 15], "point": [0, 1, 2, 4, 5, 6, 9, 10, 12, 13, 14], "pend": [0, 2, 9], "evalu": [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15], "x_pend": [0, 9], "rand": 0, "print": [0, 1, 4, 5, 7, 8, 15], "f": [0, 1, 2, 4, 5, 7, 8, 14, 15], "numpi": [0, 1, 4, 5, 7, 8, 13, 15], "reshap": [0, 1, 4, 5, 7, 8, 15], "round": [0, 1, 4, 5, 7, 8, 15], "4": [0, 1, 2, 4, 5, 7, 8, 9, 12, 14, 15], "loop": [0, 1, 2, 4, 5, 7, 8, 11, 14, 15], "iter": [0, 1, 2, 4, 5, 7, 8, 14, 15], "10": [0, 2, 7, 8, 11, 14], "rang": [0, 1, 2, 4, 5, 7, 8, 14, 15], "specifi": [0, 1, 2, 4, 5, 7, 8, 14, 15], "gaussian": [0, 2, 4, 5, 7, 8, 9, 12, 14, 15], "process": [0, 2, 4, 5, 7, 8, 9, 14, 15], "gp": [0, 1, 2, 4, 5, 7, 8, 9, 10, 15], "fit": [0, 2, 4, 5, 7, 8, 10, 14, 15], "lr": [0, 4, 5, 7, 8, 10, 11, 15], "0": [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], "step": [0, 4, 5, 7, 8, 9, 10, 11, 15], "200": [0, 4, 5, 7, 8, 10, 11, 15], "acq": [0, 1, 4, 5, 7, 8, 15], "y_best": [0, 1, 4, 7, 8, 9, 15], "max": [0, 1, 2, 4, 5, 7, 8, 15], "sampl": [0, 1, 2, 7, 8, 9, 11, 13, 14], "256": [0, 7, 8], "beta": [0, 1, 2, 4, 5, 7, 8, 9, 15], "96": [0, 1, 4, 5, 7, 8, 15], "2": [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15], "x_new": [0, 1, 4, 5, 7, 8, 15], "_": [0, 1, 2, 4, 5, 7, 8, 15], "method": [0, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14], "adam": [0, 2, 5, 7, 8, 9, 10, 11, 14], "batch_siz": [0, 7, 8, 11], "num_start": [0, 1, 4, 5, 7, 8, 11, 15], "new": [0, 1, 2, 4, 5, 7, 8, 14, 15], "y_new": [0, 1, 4, 5, 7, 8, 15], "add": [0, 1, 2, 4, 5, 7, 8, 15], "vstack": [0, 1, 4, 5, 7, 8, 15], "hstack": [0, 1, 4, 5, 7, 8, 15], "best": [0, 1, 2, 4, 5, 7, 8, 9, 11, 14, 15], "size": [0, 2, 7, 8, 9, 10, 11, 12, 13], "best_ev": [0, 7, 8], "argmax": [0, 1, 4, 5, 7, 8, 15], "t": [0, 1, 2, 4, 5, 7, 8, 15], "input": [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], "output": [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15], "result": [0, 1, 2, 4, 5, 7, 8, 11, 14, 15], "best_it": [0, 1, 4, 5, 7, 8, 15], "int": [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], "solut": [0, 1, 2, 4, 5, 7, 8, 14, 15], "float": [0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 15], "4f": [0, 1, 4, 5, 7, 8, 15], "6839": 0, "9712": 0, "6219": 0, "8499": 0, "501": [0, 15], "9095": 0, "51": [0, 4, 5, 7, 8], "3": [0, 1, 2, 4, 5, 6, 7, 8, 12, 14, 15], "881e": 0, "01": [0, 1], "6": [0, 2, 5, 12, 14], "274e": 0, "000e": 0, "04": 0, "243e": 0, "554e": 0, "4441": 0, "55": [0, 2, 5, 14, 15], "719e": 0, "8": [0, 2, 14], "601e": 0, "365e": 0, "241e": 0, "100e": 0, "03": 0, "9207": 0, "59": [0, 1, 8], "3963": 0, "9244": 0, "0608": 0, "5594": 0, "1415": 0, "0038": 0, "9958": 0, "69": [0, 7], "4065": 0, "8994": 0, "9944": 0, "5574": 0, "1096": 0, "0139": 0, "157": 0, "1570": 0, "expectedimprov": [1, 4, 9, 15], "upperconfidencebound": [1, 4, 5, 9, 15], "lbfgsb": [1, 5, 11, 15], "mll": 1, "exactmarginalloglikelihood": 1, "pyro": 1, "infer": [1, 2, 14], "mcmc": 1, "nut": 1, "constraint": [1, 2, 4, 11, 14], "posit": 1, "prior": [1, 2], "uniformprior": 1, "set": [1, 2, 5, 14], "fast_comput": 1, "seed": 1, "40": [1, 4, 5, 15], "noise_constraint": 1, "mean_modul": [1, 10], "register_prior": 1, "mean_prior": 1, "constant": [1, 2, 5, 10], "covar_modul": [1, 10], "base_kernel": 1, "lengthscale_prior": 1, "lengthscal": 1, "outputscale_prior": 1, "outputscal": 1, "noise_prior": 1, "nois": [1, 2, 12], "up": [1, 5], "def": 1, "pyro_gp": 1, "x": [1, 2, 4, 9, 10, 11, 12, 13], "y": [1, 2, 3, 10, 13], "sampled_gp": 1, "pyro_sample_from_prior": 1, "ob": 1, "return": [1, 2, 9, 10, 11, 13, 14], "run": [1, 2, 5, 14], "nuts_kernel": 1, "mcmc_run": 1, "num_sampl": [1, 11], "128": 1, "warmup_step": 1, "disable_progbar": 1, "true": [1, 2, 5, 9, 12], "load": 1, "pyro_load_from_sampl": 1, "get_sampl": 1, "lambda": [1, 4], "sum": 1, "len": [1, 4, 5, 15], "33": 1, "4456": 1, "1017": 1, "6031": 1, "9557": 1, "447": 1, "0752": 1, "351": 1, "262": 1, "3207": [1, 15], "7352": 1, "5521": 1, "45": [1, 4], "4422": 1, "9751": 1, "7018": 1, "508": 1, "0311": 1, "7326": 1, "49": 1, "1384": 1, "1482": 1, "4061": 1, "2193": 1, "2555": [1, 15], "7053": 1, "9284": 1, "1107": 1, "2156": 1, "4114": 1, "3149": 1, "297": 1, "7104": 1, "0154": 1, "62": [1, 4, 15], "4298": 1, "8893": 1, "7379": 1, "6256": 1, "166": 1, "0193": 1, "66": [1, 5], "1856": 1, "1646": 1, "4307": 1, "2932": 1, "3455": 1, "6639": [1, 4], "2274": 1, "aim": [2, 14], "solv": 2, "d": [2, 3, 9, 10, 11, 12, 13, 14], "dimension": [2, 5, 12], "boldsymbol": 2, "arg": 2, "max_": 2, "mathcal": 2, "where": [2, 6, 14], "space": [2, 11, 12, 13, 14], "i": [2, 5, 6, 9, 10, 11, 13, 14], "usual": [2, 14], "continu": [2, 14], "hyper": [2, 5, 14], "rectangl": 2, "b": [2, 5, 9, 11, 12, 14], "mathbb": 2, "r": 2, "The": [2, 5, 6, 14], "most": 2, "commonli": 2, "deriv": 2, "free": 2, "expens": [2, 3, 6, 14], "black": [2, 3, 6, 14], "box": [2, 3, 6, 14], "onli": 2, "allow": [2, 14], "x_i": 2, "querri": 2, "y_i": 2, "observ": 2, "without": 2, "gain": [2, 14], "ani": [2, 10, 11], "further": 2, "insight": 2, "underli": 2, "system": [2, 14], "we": [2, 5], "assum": 2, "epsilon": 2, "introduc": 2, "when": [2, 6], "take": [2, 14], "measur": 2, "independ": 2, "ident": 2, "distribut": [2, 5, 6, 10, 14], "sim": 2, "n": [2, 9, 10, 12, 13, 14], "sigma": 2, "henc": 2, "pair": 2, "correspond": [2, 5], "defin": [2, 4, 5, 14], "d_n": 2, "matrix": [2, 10, 14], "x_n": 2, "vector": [2, 10], "y_n": 2, "7": [2, 14], "base": [2, 9, 10, 12, 13, 14], "algorithm": [2, 5, 6, 10, 11, 14], "object": [2, 6, 12, 13, 14], "minimum": [2, 14], "number": [2, 9, 11, 12, 13, 14], "doe": [2, 14], "have": [2, 9, 14], "known": [2, 14], "mathemat": [2, 3, 14], "express": [2, 14], "everi": [2, 14], "requir": [2, 14], "cost": [2, 14], "effect": [2, 14], "effici": [2, 14], "routin": [2, 14], "meet": [2, 14], "criteria": [2, 14], "repres": [2, 14], "through": [2, 14], "m": [2, 3, 14], "often": [2, 14], "thi": [2, 5, 6, 14], "represent": [2, 14], "can": [2, 5, 6, 9, 14], "us": [2, 5, 6, 9, 10, 11, 14], "find": [2, 14], "next": [2, 14], "should": [2, 6, 9, 14], "criterion": [2, 14], "an": [2, 5, 6, 14], "alpha": 2, "popular": [2, 14], "exampl": [2, 14], "expect": [2, 9, 14], "improv": [2, 9, 14], "ei": [2, 14], "better": [2, 14], "than": [2, 14], "previou": 2, "perform": [2, 14], "_n": 2, "befor": [2, 14], "suggest": [2, 14], "ad": [2, 14], "itself": [2, 14], "restart": [2, 5, 14], "more": [2, 14], "inform": [2, 6, 14], "about": [2, 6, 14], "each": [2, 14], "mani": [2, 14], "budget": [2, 5, 14], "until": [2, 14], "satisfi": [2, 14], "found": [2, 14], "unitl": [2, 14], "pre": [2, 5, 14], "stop": [2, 14], "met": [2, 14], "initi": [2, 5, 14], "n_0": 2, "x_0": 2, "via": [2, 5, 14], "fill": [2, 14], "design": [2, 13, 14], "gather": 2, "y_0": 2, "while": [2, 14], "leq": 2, "do": 2, "increment": 2, "end": 2, "highest": 2, "choic": 2, "act": 2, "flexibl": 2, "non": 2, "parametr": 2, "regress": 2, "finit": 2, "collect": 2, "random": [2, 13, 14], "variabl": [2, 14], "ha": 2, "mean": [2, 5, 10, 13], "mu_0": 2, "mapsto": 2, "covari": [2, 10], "kernel": [2, 5, 10], "sigma_0": 2, "time": 2, "k": [2, 3, 14], "over": 2, "all": [2, 5, 6, 9, 12, 14], "between": [2, 13], "posterior": 2, "predict": 2, "n_": 2, "x_": 2, "comput": [2, 6, 9, 10, 12, 14], "multivari": [2, 10], "normal": [2, 3, 10], "condit": 2, "some": [2, 5, 6, 9, 10, 12], "mid": 2, "left": 2, "mu_n": 2, "2_n": 2, "right": 2, "paramet": [2, 5, 9, 11, 12, 13, 14], "theta": 2, "varianc": 2, "estim": [2, 5, 14], "log": 2, "margin": 2, "maximum": [2, 5, 10, 12, 14], "mle": [2, 5, 10, 14], "p": [2, 12, 14], "frac": 2, "lvert": 2, "rvert": 2, "pi": [2, 14], "packag": [2, 3, 14], "veri": 2, "power": [2, 14], "implement": [2, 5, 6, 11, 14], "wide": [2, 14], "select": [2, 13, 14], "exact": [2, 14], "approxim": [2, 14], "even": [2, 14], "deep": [2, 14], "besid": 2, "also": [2, 6, 14], "support": [2, 14], "posteriori": [2, 14], "map": [2, 14], "fulli": [2, 14], "It": [2, 6, 14], "come": 2, "rich": 2, "document": [2, 6, 14], "practic": [2, 14], "larg": [2, 13], "commun": 2, "assess": 2, "good": [2, 6], "potenti": 2, "thu": 2, "current": [2, 9, 14], "being": [2, 14], "global": [2, 12, 14], "optimum": [2, 5, 12], "To": 2, "balanc": 2, "explor": 2, "exploit": 2, "former": 2, "characteris": 2, "area": 2, "lack": 2, "uncertainti": 2, "high": 2, "latter": 2, "promis": 2, "trade": [2, 5, 9], "off": [2, 5, 9], "ensur": 2, "converg": 2, "first": [2, 5, 6], "local": 2, "full": 2, "two": 2, "ar": [2, 6, 13, 14], "ground": 2, "histori": 2, "theoret": 2, "empir": 2, "research": [2, 6, 14], "biggest": 2, "upper": [2, 5, 9, 14], "confid": [2, 5, 9, 14], "ucb": [2, 5, 14], "9": [2, 14], "optimist": 2, "view": 2, "user": 2, "level": 2, "alpha_": 2, "phi": 2, "z": [2, 14], "sigma_n": 2, "cdot": 2, "standard": [2, 12, 13], "deviat": [2, 12, 13], "cumul": 2, "probabl": 2, "densiti": 2, "sqrt": 2, "both": 2, "them": [2, 14], "determinist": [2, 9, 14], "l": [2, 5, 9, 11, 14], "bfg": [2, 5, 9, 11, 14], "unconstraint": 2, "slsqp": [2, 4, 9, 11, 14], "howev": 2, "sequenti": [2, 6, 11, 14], "singl": [2, 6, 14], "case": [2, 5], "which": [2, 5, 11], "immediatlei": 2, "repeat": 2, "contain": [2, 6, 12], "matern": [2, 5, 10], "especi": 2, "suit": 2, "For": [2, 5], "parallel": [2, 6, 14], "multi": [2, 5, 6, 14], "batch": [2, 11, 14], "asynchron": [2, 6, 14], "gener": [2, 5, 6, 11, 14], "intract": 2, "11": [2, 14], "idea": 2, "draw": [2, 11, 13], "directli": [2, 5], "predicitv": 2, "averag": [2, 9], "made": 2, "viabl": 2, "reparameteris": 2, "utilis": 2, "mc": 2, "relu": 2, "lower": 2, "triangular": 2, "choleski": 2, "decomposit": 2, "rectifi": 2, "linear": 2, "unit": [2, 13], "zero": [2, 10], "valu": [2, 14], "below": 2, "leav": 2, "rest": 2, "due": 2, "stochast": [2, 9, 14], "evid": 2, "fix": [2, 9], "individu": 2, "affect": 2, "neg": [2, 9], "would": 2, "could": 2, "bia": 2, "furthermor": 2, "strategi": 2, "possibl": 2, "default": [2, 9, 10, 11, 13], "approach": [2, 5], "second": 2, "option": [2, 10, 11, 13], "greedi": [2, 11], "one": [2, 13], "after": [2, 14], "other": [2, 14], "hold": 2, "show": 2, "similarli": 2, "smaller": 2, "larger": 2, "increas": 2, "complex": 2, "leverag": 2, "same": 2, "properti": 2, "yet": 2, "been": 2, "treat": 2, "In": [2, 5], "wai": 2, "thei": 2, "consid": [2, 13], "figur": 2, "three": [2, 14], "dark": 2, "blue": 2, "dot": 2, "solid": 2, "red": 2, "line": 2, "here": 2, "95": [2, 5], "interv": [2, 5], "orang": 2, "dash": 2, "onc": 2, "final": [2, 5], "compar": 2, "last": 2, "frazier": [2, 14], "tutori": [2, 14], "optim": [2, 3, 9, 10, 11, 14], "arxiv": [2, 3, 14], "preprint": [2, 3, 14], "1807": [2, 14], "02811": [2, 14], "2018": [2, 14], "j": [2, 3, 14], "gardner": [2, 14], "g": [2, 14], "pleiss": [2, 14], "kq": [2, 14], "weinberg": [2, 14], "bindel": [2, 14], "ag": [2, 14], "wilson": [2, 3, 14], "blackbox": [2, 14], "gpu": [2, 14], "acceler": [2, 14], "advanc": [2, 14], "neural": [2, 14], "vol": [2, 14], "31": [2, 14], "rb": [2, 14], "gramaci": [2, 14], "appli": [2, 3, 14], "scienc": [2, 14], "1st": [2, 14], "ed": [2, 14], "boca": [2, 14], "raton": [2, 14], "fl": [2, 14], "crc": [2, 14], "press": [2, 14], "2020": [2, 14], "dr": [2, 14], "jone": [2, 14], "schonlau": [2, 14], "wj": [2, 14], "welch": [2, 14], "journal": [2, 3, 14], "13": [2, 14], "566": [2, 14], "1998": [2, 14], "dp": [2, 14], "kingma": [2, 14], "ba": [2, 14], "proceed": [2, 14], "3rd": [2, 14], "intern": [2, 14], "confer": [2, 14], "learn": [2, 10, 11, 14], "2015": [2, 14], "md": [2, 14], "mckai": [2, 14], "rj": [2, 14], "beckman": [2, 14], "conov": [2, 14], "comparison": [2, 14], "analysi": [2, 3, 14], "code": [2, 5, 6, 14], "technometr": [2, 14], "42": [2, 14, 15], "61": [2, 14], "2000": [2, 14], "shahriari": [2, 14], "swerski": [2, 14], "wang": [2, 14], "rp": [2, 14], "de": [2, 14], "freita": [2, 14], "human": [2, 14], "out": [2, 14], "review": [2, 14], "ieee": [2, 14], "104": [2, 14], "148": [2, 14], "175": [2, 14], "snoek": [2, 14], "h": [2, 14], "larochel": [2, 14], "machin": [2, 14], "25": [2, 14], "2012": [2, 14], "sriniva": [2, 14], "kraus": [2, 14], "sm": [2, 14], "kakad": [2, 14], "seeger": [2, 14], "bandit": [2, 14], "No": [2, 14], "regret": [2, 14], "experiment": [2, 6, 14], "27th": [2, 14], "1015": [2, 14], "1022": [2, 14], "2010": [2, 14], "cki": [2, 14], "william": [2, 14], "ce": [2, 14], "rasmussen": [2, 14], "2nd": [2, 14], "cambridg": [2, 14], "ma": [2, 14], "mit": [2, 14], "2006": [2, 14], "hutter": [2, 14], "deisenroth": [2, 14], "maxim": [2, 14], "pleas": 3, "cite": 3, "diessner": 3, "rd": 3, "whallei": 3, "transpar": [3, 14], "python": [3, 14], "bayesian": [3, 5], "optimis": [3, 9, 10], "1234": 3, "56789": 3, "2023": [3, 14], "bibtex": 3, "articl": 3, "nubo2023": 3, "titl": 3, "author": 3, "mike": 3, "kevin": 3, "richard": 3, "year": 3, "o": 3, "connor": 3, "A": [3, 6, 12, 14], "wynn": 3, "": [3, 6, 14], "laizet": 3, "streamwis": 3, "vari": 3, "wall": 3, "blow": 3, "turbul": 3, "boundari": 3, "layer": 3, "flow": 3, "combust": 3, "guan": 3, "investig": 3, "applic": 3, "fluid": [3, 6, 14], "dynam": [3, 6, 14], "frontier": 3, "statist": 3, "2022": 3, "con": 4, "type": 4, "ineq": 4, "fun": 4, "eq": 4, "2442": 4, "37": 4, "2957": 4, "2043": 4, "5873": 4, "2205": 4, "2488": 4, "7749": 4, "5862": 4, "41": [4, 15], "2762": 4, "2129": 4, "5612": 4, "2206": 4, "2766": 4, "747": 4, "8477": 4, "2587": 4, "5491": 4, "2672": 4, "274": 4, "703": 4, "867": 4, "50": [4, 5], "2671": 4, "2329": 4, "52": [4, 5], "2676": 4, "2833": 4, "6934": 4, "1119": 4, "2393": 4, "1902": 4, "4034": 4, "2632": 4, "3038": 4, "6772": 4, "2298": 4, "54": [4, 5], "2175": 4, "1541": 4, "4372": 4, "285": 4, "3091": 4, "6501": 4, "3029": 4, "60": 4, "2094": 4, "1434": 4, "486": 4, "2706": 4, "3062": 4, "6675": 4, "3153": 4, "2042": 4, "1543": 4, "484": 4, "2716": 4, "3087": 4, "3197": 4, "63": [4, 15], "2046": 4, "1547": 4, "4831": 4, "2726": 4, "6625": 4, "3204": [4, 5], "64": [4, 5, 7], "2044": 4, "1565": 4, "4828": 4, "3093": [4, 15], "6609": 4, "3208": 4, "65": 4, "2038": 4, "1558": 4, "4806": 4, "2732": 4, "3105": [4, 15], "6605": 4, "3213": [4, 15], "its": [5, 6], "depend": 5, "github": [5, 6], "repositori": 5, "pip": 5, "follow": 5, "virtual": [5, 14], "environ": 5, "recommend": 5, "git": 5, "http": [5, 14], "com": 5, "mikediessn": 5, "want": [5, 6], "choos": 5, "hartmann": 5, "modal": 5, "Then": 5, "decid": 5, "per": 5, "dimens": [5, 9, 12, 13], "total": [5, 11], "30": 5, "now": 5, "prepar": 5, "analyt": [5, 14], "five": 5, "give": [5, 6], "70": 5, "43": [5, 8], "18": 5, "147": 5, "1909": 5, "3424": 5, "7121": 5, "1026": 5, "46": 5, "2992": 5, "1852": 5, "21": 5, "3398": 5, "6985": 5, "48": [5, 15], "2597": 5, "1744": 5, "2323": 5, "3173": 5, "6544": 5, "337": 5, "2486": 5, "1728": 5, "112": 5, "2413": 5, "2927": 5, "6674": 5, "6599": [5, 15], "234": 5, "1519": 5, "2624": 5, "2972": 5, "6662": 5, "1372": 5, "2117": 5, "1087": 5, "3731": 5, "313": 5, "3146": [5, 7], "1906": 5, "1698": 5, "1394": 5, "405": 5, "3109": 5, "2839": 5, "6623": 5, "1964": 5, "1431": 5, "1126": 5, "4022": 5, "2795": 5, "3051": 5, "635": 5, "198": 5, "58": 5, "2112": 5, "1557": 5, "4745": 5, "288": 5, "3086": 5, "6555": 5, "3158": 5, "2013": 5, "1443": 5, "4779": 5, "2734": [5, 15], "3131": 5, "6584": 5, "3218": 5, "overal": 5, "approximati": 5, "3224": 5, "short": [6, 14], "newcastl": [6, 14], "univers": [6, 14], "framework": [6, 14], "physic": [6, 14], "experi": [6, 14], "simul": [6, 14], "develop": [6, 14], "group": [6, 14], "focus": [6, 14], "precis": [6, 14], "make": [6, 14], "access": [6, 14], "disciplin": [6, 14], "written": [6, 14], "open": [6, 14], "sourc": [6, 9, 10, 11, 12, 13, 14], "under": [6, 14], "bsd": [6, 14], "claus": [6, 14], "licens": [6, 14], "section": 6, "depth": 6, "explan": 6, "compon": 6, "surrog": [6, 14], "quickstart": 6, "guid": 6, "so": 6, "you": 6, "start": [6, 11, 14], "your": 6, "minut": 6, "place": 6, "journei": 6, "overview": 6, "get": 6, "primer": 6, "citat": 6, "provid": [6, 13, 14], "problem": [6, 12, 14], "capabl": 6, "boilerpl": 6, "tailor": 6, "specfic": 6, "constrain": 6, "detail": 6, "go": 6, "sure": 6, "how": 6, "specif": 6, "modul": 6, "512": [7, 8, 9], "2378": 7, "1476": 7, "4675": 7, "2661": 7, "3609": 7, "6194": 7, "1505": 7, "57": 7, "2053": [7, 15], "1013": 7, "4377": 7, "2808": 7, "6717": 7, "2772": 7, "2143": 7, "1669": 7, "4547": 7, "2821": 7, "303": 7, "654": 7, "3085": 7, "2118": 7, "1493": 7, "4875": 7, "2634": 7, "3155": 7, "6649": 7, "3135": 7, "35": [8, 15], "3297": [8, 15], "0108": 8, "1252": 8, "1826": 8, "1695": 8, "7313": 8, "5985": 8, "39": 8, "0422": 8, "0106": 8, "0614": 8, "2066": 8, "2079": 8, "749": 8, "6499": 8, "2429": 8, "0024": 8, "1681": 8, "235": 8, "196": 8, "6835": 8, "1151": 8, "47": [8, 15], "257": 8, "0045": 8, "2255": 8, "2424": 8, "2303": 8, "6589": 8, "4768": [8, 15], "2588": 8, "0019": 8, "3118": 8, "2619": 8, "2567": 8, "6526": 8, "777": 8, "2349": 8, "0065": 8, "3678": 8, "3052": 8, "2844": 8, "6723": 8, "9674": 8, "67": 8, "2252": 8, "0888": 8, "367": 8, "2958": 8, "6721": 8, "1527": 8, "acquisition_funct": 9, "acquisitionfunct": 9, "tensor": [9, 10, 11, 12, 13], "attribut": [9, 10, 12, 13], "eval": [9, 12], "imrpov": 9, "none": [9, 10, 11, 12, 13], "monte_carlo": 9, "fix_base_sampl": 9, "bool": [9, 12], "whether": 9, "If": [9, 13], "base_sampl": 9, "nonetyp": 9, "drawn": 9, "class": [10, 13], "gaussian_process": 10, "exactgp": 10, "automat": 10, "relev": 10, "determin": 10, "forward": 10, "multivariatenorm": 10, "predictic": 10, "kwarg": [10, 11], "target": 10, "rate": [10, 11], "keyword": [10, 11], "argument": [10, 11], "pass": [10, 11], "callabl": 11, "100": 11, "tupl": 11, "scipi": 11, "minim": [11, 13], "pick": 11, "latin": [11, 13, 14], "hypercub": [11, 13, 14], "initialis": 11, "optims": 11, "best_result": 11, "best_func_result": 11, "dict": [11, 12], "list": 11, "pytorch": 11, "multipoint": 11, "str": 11, "mont": [11, 14], "carlo": [11, 14], "One": 11, "minimz": 11, "batch_result": 11, "sizq": 11, "batch_func_result": 11, "gen_candid": 11, "num_candid": 11, "candid": 11, "testfunct": 12, "noise_std": 12, "maximis": [12, 14], "c": 12, "dixonpric": 12, "hartmann3d": 12, "sumsquar": 12, "latin_hypercub": 13, "latinhypercubesampl": 13, "maximin": [13, 14], "1000": 13, "largest": 13, "distanc": 13, "ndarrai": 13, "generate_input": 13, "num_lhd": 13, "standardis": 13, "subtract": 13, "divid": 13, "normalis": 13, "cube": 13, "unnormalis": 13, "rever": 13, "scale": 13, "refer": 14, "optimnis": 14, "still": 14, "12": 14, "restrict": 14, "synthet": 14, "ten": 14, "valid": 14, "surjanov": 14, "bingham": 14, "librari": 14, "dataset": 14, "sfu": 14, "ca": 14, "onlin": 14, "avail": 14, "www": 14, "ssurjano": 14, "html": 14, "march": 14, "34": 15, "1944": 15, "2627": 15, "311": 15, "3311": 15, "7252": 15, "2019": 15, "2237": 15, "1796": 15, "3295": 15, "3233": 15, "6819": 15, "3608": 15, "36": 15, "2334": 15, "1673": 15, "3209": 15, "6827": 15, "194": 15, "2061": 15, "1872": 15, "513": 15, "317": 15, "2864": 15, "642": 15, "2033": 15, "1491": 15, "4892": 15, "2897": 15, "2878": 15, "6616": 15, "2528": 15, "1926": 15, "161": 15, "5281": 15, "2637": 15, "3057": 15, "683": 15, "2236": 15, "1315": 15, "4595": 15, "2746": 15, "3192": 15, "6622": 15, "3069": 15, "1971": 15, "1513": 15, "4775": 15, "278": 15, "3096": 15, "6462": 15, "3184": 15, "53": 15, "207": 15, "1481": 15, "4719": 15, "2767": 15, "6667": 15, "3193": 15, "2069": 15, "1546": 15, "4753": 15, "2798": 15, "3106": 15, "6618": 15, "2056": 15, "154": 15, "4771": 15, "2802": 15, "6606": 15, "321": 15, "1533": 15, "2799": 15}, "objects": {"nubo.acquisition": [[9, 0, 0, "-", "acquisition_function"], [9, 0, 0, "-", "analytical"], [9, 0, 0, "-", "monte_carlo"]], "nubo.acquisition.acquisition_function": [[9, 1, 1, "", "AcquisitionFunction"]], "nubo.acquisition.analytical": [[9, 1, 1, "", "ExpectedImprovement"], [9, 1, 1, "", "UpperConfidenceBound"]], "nubo.acquisition.analytical.ExpectedImprovement": [[9, 2, 1, "", "eval"]], "nubo.acquisition.analytical.UpperConfidenceBound": [[9, 2, 1, "", "eval"]], "nubo.acquisition.monte_carlo": [[9, 1, 1, "", "MCExpectedImprovement"], [9, 1, 1, "", "MCUpperConfidenceBound"]], "nubo.acquisition.monte_carlo.MCExpectedImprovement": [[9, 2, 1, "", "eval"]], "nubo.acquisition.monte_carlo.MCUpperConfidenceBound": [[9, 2, 1, "", "eval"]], "nubo.models": [[10, 0, 0, "-", "fit"], [10, 0, 0, "-", "gaussian_process"]], "nubo.models.fit": [[10, 3, 1, "", "fit_gp"]], "nubo.models.gaussian_process": [[10, 1, 1, "", "GaussianProcess"]], "nubo.models.gaussian_process.GaussianProcess": [[10, 2, 1, "", "forward"]], "nubo.optimisation": [[11, 0, 0, "-", "adam"], [11, 0, 0, "-", "lbfgsb"], [11, 0, 0, "-", "multipoint"], [11, 0, 0, "-", "slsqp"], [11, 0, 0, "-", "utils"]], "nubo.optimisation.adam": [[11, 3, 1, "", "adam"]], "nubo.optimisation.lbfgsb": [[11, 3, 1, "", "lbfgsb"]], "nubo.optimisation.multipoint": [[11, 3, 1, "", "joint"], [11, 3, 1, "", "sequential"]], "nubo.optimisation.slsqp": [[11, 3, 1, "", "slsqp"]], "nubo.optimisation.utils": [[11, 3, 1, "", "gen_candidates"]], "nubo.test_functions": [[12, 0, 0, "-", "ackley"], [12, 0, 0, "-", "dixonprice"], [12, 0, 0, "-", "griewank"], [12, 0, 0, "-", "hartmann"], [12, 0, 0, "-", "levy"], [12, 0, 0, "-", "rastrigin"], [12, 0, 0, "-", "schwefel"], [12, 0, 0, "-", "sphere"], [12, 0, 0, "-", "sumsquares"], [12, 0, 0, "-", "test_functions"], [12, 0, 0, "-", "zakharov"]], "nubo.test_functions.ackley": [[12, 1, 1, "", "Ackley"]], "nubo.test_functions.ackley.Ackley": [[12, 2, 1, "", "eval"]], "nubo.test_functions.dixonprice": [[12, 1, 1, "", "DixonPrice"]], "nubo.test_functions.dixonprice.DixonPrice": [[12, 2, 1, "", "eval"]], "nubo.test_functions.griewank": [[12, 1, 1, "", "Griewank"]], "nubo.test_functions.griewank.Griewank": [[12, 2, 1, "", "eval"]], "nubo.test_functions.hartmann": [[12, 1, 1, "", "Hartmann3D"], [12, 1, 1, "", "Hartmann6D"]], "nubo.test_functions.hartmann.Hartmann3D": [[12, 2, 1, "", "eval"]], "nubo.test_functions.hartmann.Hartmann6D": [[12, 2, 1, "", "eval"]], "nubo.test_functions.levy": [[12, 1, 1, "", "Levy"]], "nubo.test_functions.levy.Levy": [[12, 2, 1, "", "eval"]], "nubo.test_functions.rastrigin": [[12, 1, 1, "", "Rastrigin"]], "nubo.test_functions.rastrigin.Rastrigin": [[12, 2, 1, "", "eval"]], "nubo.test_functions.schwefel": [[12, 1, 1, "", "Schwefel"]], "nubo.test_functions.schwefel.Schwefel": [[12, 2, 1, "", "eval"]], "nubo.test_functions.sphere": [[12, 1, 1, "", "Sphere"]], "nubo.test_functions.sphere.Sphere": [[12, 2, 1, "", "eval"]], "nubo.test_functions.sumsquares": [[12, 1, 1, "", "SumSquares"]], "nubo.test_functions.sumsquares.SumSquares": [[12, 2, 1, "", "eval"]], "nubo.test_functions.test_functions": [[12, 1, 1, "", "TestFunction"]], "nubo.test_functions.zakharov": [[12, 1, 1, "", "Zakharov"]], "nubo.test_functions.zakharov.Zakharov": [[12, 2, 1, "", "eval"]], "nubo.utils": [[13, 0, 0, "-", "generate_inputs"], [13, 0, 0, "-", "latin_hypercube"], [13, 0, 0, "-", "transform"]], "nubo.utils.generate_inputs": [[13, 3, 1, "", "gen_inputs"]], "nubo.utils.latin_hypercube": [[13, 1, 1, "", "LatinHypercubeSampling"]], "nubo.utils.latin_hypercube.LatinHypercubeSampling": [[13, 2, 1, "", "maximin"], [13, 2, 1, "", "random"]], "nubo.utils.transform": [[13, 3, 1, "", "normalise"], [13, 3, 1, "", "standardise"], [13, 3, 1, "", "unnormalise"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"asynchron": 0, "bayesian": [0, 1, 2, 4, 6, 7, 8, 14, 15], "optimis": [0, 1, 2, 4, 5, 6, 7, 8, 11, 14, 15], "fulli": 1, "gaussian": [1, 10], "process": [1, 10], "A": 2, "primer": 2, "maximis": 2, "problem": 2, "surrog": [2, 10], "model": 2, "acquisit": [2, 9], "function": [2, 5, 9, 12], "analyt": [2, 9], "mont": [2, 9], "carlo": [2, 9], "citat": 3, "select": 3, "public": 3, "us": 3, "nubo": [3, 5, 6], "constrain": 4, "get": 5, "start": 5, "instal": 5, "toi": 5, "transpar": 6, "python": 6, "packag": 6, "exampl": 6, "refer": 6, "parallel": [7, 8], "multi": [7, 8, 11], "point": [7, 8, 11, 15], "joint": 7, "sequenti": [8, 15], "modul": [9, 10, 11, 12, 13], "parent": [9, 12], "class": [9, 12], "aquisit": 9, "hyper": 10, "paramet": 10, "estim": 10, "determinist": 11, "stochast": 11, "util": [11, 13], "test": 12, "acklei": 12, "dixon": 12, "price": 12, "griewank": 12, "hartmann": 12, "levi": 12, "rastrigin": 12, "schwefel": 12, "sphere": 12, "sum": 12, "squar": 12, "zakharov": 12, "data": 13, "gener": 13, "transform": 13, "overview": 14, "content": 14, "singl": 15}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Asynchronous Bayesian optimisation": [[0, "Asynchronous-Bayesian-optimisation"]], "Fully Bayesian Gaussian process for Bayesian optimisation": [[1, "Fully-Bayesian-Gaussian-process-for-Bayesian-optimisation"]], "A primer on Bayesian optimisation": [[2, "a-primer-on-bayesian-optimisation"]], "Maximisation problem": [[2, "maximisation-problem"]], "Bayesian optimisation": [[2, "bayesian-optimisation"], [14, "bayesian-optimisation"]], "Surrogate model": [[2, "surrogate-model"]], "Acquisition function": [[2, "acquisition-function"]], "Analytical acquisition functions": [[2, "analytical-acquisition-functions"], [9, "module-nubo.acquisition.analytical"]], "Monte Carlo acquisition functions": [[2, "monte-carlo-acquisition-functions"]], "Citation": [[3, "citation"]], "Selected publications using NUBO": [[3, "selected-publications-using-nubo"]], "Constrained Bayesian optimisation": [[4, "Constrained-Bayesian-optimisation"]], "Get started": [[5, "Get-started"]], "Installing NUBO": [[5, "Installing-NUBO"]], "Optimising a toy function with NUBO": [[5, "Optimising-a-toy-function-with-NUBO"]], "NUBO: a transparent python package for Bayesian optimisation": [[6, "nubo-a-transparent-python-package-for-bayesian-optimisation"]], "NUBO": [[6, "nubo"]], "NUBO:": [[6, null]], "Examples": [[6, "examples"]], "Examples:": [[6, null]], "Package reference": [[6, "package-reference"]], "Package reference:": [[6, null]], "Parallel multi-point joint Bayesian optimisation": [[7, "Parallel-multi-point-joint-Bayesian-optimisation"]], "Parallel multi-point sequential Bayesian optimisation": [[8, "Parallel-multi-point-sequential-Bayesian-optimisation"]], "Acquisition module": [[9, "acquisition-module"]], "Parent class": [[9, "module-nubo.acquisition.acquisition_function"], [12, "module-nubo.test_functions.test_functions"]], "Monte Carlo aquisition functions": [[9, "module-nubo.acquisition.monte_carlo"]], "Surrogates module": [[10, "surrogates-module"]], "Gaussian Process": [[10, "module-nubo.models.gaussian_process"]], "Hyper-parameter estimation": [[10, "module-nubo.models.fit"]], "Optimisation module": [[11, "optimisation-module"]], "Deterministic optimisers": [[11, "module-nubo.optimisation.lbfgsb"]], "Stochastic optimisers": [[11, "module-nubo.optimisation.adam"]], "Multi-point optimisation": [[11, "module-nubo.optimisation.multipoint"]], "Optimisation utilities": [[11, "module-nubo.optimisation.utils"]], "Test function module": [[12, "test-function-module"]], "Ackley function": [[12, "module-nubo.test_functions.ackley"]], "Dixon-Price function": [[12, "module-nubo.test_functions.dixonprice"]], "Griewank function": [[12, "module-nubo.test_functions.griewank"]], "Hartmann function": [[12, "module-nubo.test_functions.hartmann"]], "Levy function": [[12, "module-nubo.test_functions.levy"]], "Rastrigin function": [[12, "module-nubo.test_functions.rastrigin"]], "Schwefel function": [[12, "module-nubo.test_functions.schwefel"]], "Sphere function": [[12, "module-nubo.test_functions.sphere"]], "Sum-of-Squares function": [[12, "module-nubo.test_functions.sumsquares"]], "Zakharov function": [[12, "module-nubo.test_functions.zakharov"]], "Utility module": [[13, "utility-module"]], "Data generation": [[13, "module-nubo.utils.latin_hypercube"]], "Data transformations": [[13, "module-nubo.utils.transform"]], "Overview": [[14, "overview"]], "Contents": [[14, "contents"]], "Sequential single-point Bayesian optimisation": [[15, "Sequential-single-point-Bayesian-optimisation"]]}, "indexentries": {"acquisitionfunction (class in nubo.acquisition.acquisition_function)": [[9, "nubo.acquisition.acquisition_function.AcquisitionFunction"]], "expectedimprovement (class in nubo.acquisition.analytical)": [[9, "nubo.acquisition.analytical.ExpectedImprovement"]], "mcexpectedimprovement (class in nubo.acquisition.monte_carlo)": [[9, "nubo.acquisition.monte_carlo.MCExpectedImprovement"]], "mcupperconfidencebound (class in nubo.acquisition.monte_carlo)": [[9, "nubo.acquisition.monte_carlo.MCUpperConfidenceBound"]], "upperconfidencebound (class in nubo.acquisition.analytical)": [[9, "nubo.acquisition.analytical.UpperConfidenceBound"]], "eval() (nubo.acquisition.analytical.expectedimprovement method)": [[9, "nubo.acquisition.analytical.ExpectedImprovement.eval"]], "eval() (nubo.acquisition.analytical.upperconfidencebound method)": [[9, "nubo.acquisition.analytical.UpperConfidenceBound.eval"]], "eval() (nubo.acquisition.monte_carlo.mcexpectedimprovement method)": [[9, "nubo.acquisition.monte_carlo.MCExpectedImprovement.eval"]], "eval() (nubo.acquisition.monte_carlo.mcupperconfidencebound method)": [[9, "nubo.acquisition.monte_carlo.MCUpperConfidenceBound.eval"]], "module": [[9, "module-nubo.acquisition.acquisition_function"], [9, "module-nubo.acquisition.analytical"], [9, "module-nubo.acquisition.monte_carlo"], [10, "module-nubo.models.fit"], [10, "module-nubo.models.gaussian_process"], [11, "module-nubo.optimisation.adam"], [11, "module-nubo.optimisation.lbfgsb"], [11, "module-nubo.optimisation.multipoint"], [11, "module-nubo.optimisation.slsqp"], [11, "module-nubo.optimisation.utils"], [12, "module-nubo.test_functions.ackley"], [12, "module-nubo.test_functions.dixonprice"], [12, "module-nubo.test_functions.griewank"], [12, "module-nubo.test_functions.hartmann"], [12, "module-nubo.test_functions.levy"], [12, "module-nubo.test_functions.rastrigin"], [12, "module-nubo.test_functions.schwefel"], [12, "module-nubo.test_functions.sphere"], [12, "module-nubo.test_functions.sumsquares"], [12, "module-nubo.test_functions.test_functions"], [12, "module-nubo.test_functions.zakharov"], [13, "module-nubo.utils.generate_inputs"], [13, "module-nubo.utils.latin_hypercube"], [13, "module-nubo.utils.transform"]], "nubo.acquisition.acquisition_function": [[9, "module-nubo.acquisition.acquisition_function"]], "nubo.acquisition.analytical": [[9, "module-nubo.acquisition.analytical"]], "nubo.acquisition.monte_carlo": [[9, "module-nubo.acquisition.monte_carlo"]], "gaussianprocess (class in nubo.models.gaussian_process)": [[10, "nubo.models.gaussian_process.GaussianProcess"]], "fit_gp() (in module nubo.models.fit)": [[10, "nubo.models.fit.fit_gp"]], "forward() (nubo.models.gaussian_process.gaussianprocess method)": [[10, "nubo.models.gaussian_process.GaussianProcess.forward"]], "nubo.models.fit": [[10, "module-nubo.models.fit"]], "nubo.models.gaussian_process": [[10, "module-nubo.models.gaussian_process"]], "adam() (in module nubo.optimisation.adam)": [[11, "nubo.optimisation.adam.adam"]], "gen_candidates() (in module nubo.optimisation.utils)": [[11, "nubo.optimisation.utils.gen_candidates"]], "joint() (in module nubo.optimisation.multipoint)": [[11, "nubo.optimisation.multipoint.joint"]], "lbfgsb() (in module nubo.optimisation.lbfgsb)": [[11, "nubo.optimisation.lbfgsb.lbfgsb"]], "nubo.optimisation.adam": [[11, "module-nubo.optimisation.adam"]], "nubo.optimisation.lbfgsb": [[11, "module-nubo.optimisation.lbfgsb"]], "nubo.optimisation.multipoint": [[11, "module-nubo.optimisation.multipoint"]], "nubo.optimisation.slsqp": [[11, "module-nubo.optimisation.slsqp"]], "nubo.optimisation.utils": [[11, "module-nubo.optimisation.utils"]], "sequential() (in module nubo.optimisation.multipoint)": [[11, "nubo.optimisation.multipoint.sequential"]], "slsqp() (in module nubo.optimisation.slsqp)": [[11, "nubo.optimisation.slsqp.slsqp"]], "ackley (class in nubo.test_functions.ackley)": [[12, "nubo.test_functions.ackley.Ackley"]], "dixonprice (class in nubo.test_functions.dixonprice)": [[12, "nubo.test_functions.dixonprice.DixonPrice"]], "griewank (class in nubo.test_functions.griewank)": [[12, "nubo.test_functions.griewank.Griewank"]], "hartmann3d (class in nubo.test_functions.hartmann)": [[12, "nubo.test_functions.hartmann.Hartmann3D"]], "hartmann6d (class in nubo.test_functions.hartmann)": [[12, "nubo.test_functions.hartmann.Hartmann6D"]], "levy (class in nubo.test_functions.levy)": [[12, "nubo.test_functions.levy.Levy"]], "rastrigin (class in nubo.test_functions.rastrigin)": [[12, "nubo.test_functions.rastrigin.Rastrigin"]], "schwefel (class in nubo.test_functions.schwefel)": [[12, "nubo.test_functions.schwefel.Schwefel"]], "sphere (class in nubo.test_functions.sphere)": [[12, "nubo.test_functions.sphere.Sphere"]], "sumsquares (class in nubo.test_functions.sumsquares)": [[12, "nubo.test_functions.sumsquares.SumSquares"]], "testfunction (class in nubo.test_functions.test_functions)": [[12, "nubo.test_functions.test_functions.TestFunction"]], "zakharov (class in nubo.test_functions.zakharov)": [[12, "nubo.test_functions.zakharov.Zakharov"]], "eval() (nubo.test_functions.ackley.ackley method)": [[12, "nubo.test_functions.ackley.Ackley.eval"]], "eval() (nubo.test_functions.dixonprice.dixonprice method)": [[12, "nubo.test_functions.dixonprice.DixonPrice.eval"]], "eval() (nubo.test_functions.griewank.griewank method)": [[12, "nubo.test_functions.griewank.Griewank.eval"]], "eval() (nubo.test_functions.hartmann.hartmann3d method)": [[12, "nubo.test_functions.hartmann.Hartmann3D.eval"]], "eval() (nubo.test_functions.hartmann.hartmann6d method)": [[12, "nubo.test_functions.hartmann.Hartmann6D.eval"]], "eval() (nubo.test_functions.levy.levy method)": [[12, "nubo.test_functions.levy.Levy.eval"]], "eval() (nubo.test_functions.rastrigin.rastrigin method)": [[12, "nubo.test_functions.rastrigin.Rastrigin.eval"]], "eval() (nubo.test_functions.schwefel.schwefel method)": [[12, "nubo.test_functions.schwefel.Schwefel.eval"]], "eval() (nubo.test_functions.sphere.sphere method)": [[12, "nubo.test_functions.sphere.Sphere.eval"]], "eval() (nubo.test_functions.sumsquares.sumsquares method)": [[12, "nubo.test_functions.sumsquares.SumSquares.eval"]], "eval() (nubo.test_functions.zakharov.zakharov method)": [[12, "nubo.test_functions.zakharov.Zakharov.eval"]], "nubo.test_functions.ackley": [[12, "module-nubo.test_functions.ackley"]], "nubo.test_functions.dixonprice": [[12, "module-nubo.test_functions.dixonprice"]], "nubo.test_functions.griewank": [[12, "module-nubo.test_functions.griewank"]], "nubo.test_functions.hartmann": [[12, "module-nubo.test_functions.hartmann"]], "nubo.test_functions.levy": [[12, "module-nubo.test_functions.levy"]], "nubo.test_functions.rastrigin": [[12, "module-nubo.test_functions.rastrigin"]], "nubo.test_functions.schwefel": [[12, "module-nubo.test_functions.schwefel"]], "nubo.test_functions.sphere": [[12, "module-nubo.test_functions.sphere"]], "nubo.test_functions.sumsquares": [[12, "module-nubo.test_functions.sumsquares"]], "nubo.test_functions.test_functions": [[12, "module-nubo.test_functions.test_functions"]], "nubo.test_functions.zakharov": [[12, "module-nubo.test_functions.zakharov"]], "latinhypercubesampling (class in nubo.utils.latin_hypercube)": [[13, "nubo.utils.latin_hypercube.LatinHypercubeSampling"]], "gen_inputs() (in module nubo.utils.generate_inputs)": [[13, "nubo.utils.generate_inputs.gen_inputs"]], "maximin() (nubo.utils.latin_hypercube.latinhypercubesampling method)": [[13, "nubo.utils.latin_hypercube.LatinHypercubeSampling.maximin"]], "normalise() (in module nubo.utils.transform)": [[13, "nubo.utils.transform.normalise"]], "nubo.utils.generate_inputs": [[13, "module-nubo.utils.generate_inputs"]], "nubo.utils.latin_hypercube": [[13, "module-nubo.utils.latin_hypercube"]], "nubo.utils.transform": [[13, "module-nubo.utils.transform"]], "random() (nubo.utils.latin_hypercube.latinhypercubesampling method)": [[13, "nubo.utils.latin_hypercube.LatinHypercubeSampling.random"]], "standardise() (in module nubo.utils.transform)": [[13, "nubo.utils.transform.standardise"]], "unnormalise() (in module nubo.utils.transform)": [[13, "nubo.utils.transform.unnormalise"]]}})