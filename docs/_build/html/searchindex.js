Search.setIndex({"docnames": ["asynchronous_bo", "bayesian_gp", "bayesian_optimisation", "citation", "constrained_bo", "get_started", "index", "multipoint_joint", "multipoint_sequential", "nubo.acquisition", "nubo.models", "nubo.optimisation", "nubo.test_functions", "nubo.utils", "overview", "singlepoint"], "filenames": ["asynchronous_bo.ipynb", "bayesian_gp.ipynb", "bayesian_optimisation.rst", "citation.md", "constrained_bo.ipynb", "get_started.ipynb", "index.rst", "multipoint_joint.ipynb", "multipoint_sequential.ipynb", "nubo.acquisition.rst", "nubo.models.rst", "nubo.optimisation.rst", "nubo.test_functions.rst", "nubo.utils.rst", "overview.rst", "singlepoint.ipynb"], "titles": ["Asynchronous Bayesian Optimisation", "Fully Bayesian Gaussian Process for Bayesian Optimisation", "Primer on Bayesian optimisation", "Citation", "Constrained Bayesian Optimisation", "Get started", "NUBO: a transparent python package for Bayesian optimisation", "Parallel multi-point joint Bayesian Optimisation", "Parallel multi-point sequential Bayesian Optimisation", "Acquisition module", "Surrogates module", "Optimisation module", "Test function module", "Utility module", "Overview", "Sequential single-point Bayesian Optimisation"], "terms": {"1": [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], "import": [1, 4, 5, 7, 8, 15], "torch": [1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], "from": [1, 2, 4, 5, 6, 7, 8, 11, 14, 15], "nubo": [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15], "acquisit": [1, 4, 5, 6, 7, 8, 11, 14, 15], "expectedimprov": [1, 4, 9, 15], "upperconfidencebound": [1, 4, 5, 9, 15], "model": [1, 4, 5, 6, 7, 8, 9, 10, 14, 15], "gaussianprocess": [1, 4, 5, 7, 8, 10, 15], "lbfgsb": [1, 5, 11, 15], "test_funct": [1, 4, 5, 7, 8, 12, 15], "hartmann6d": [1, 4, 5, 7, 8, 12, 15], "util": [1, 4, 5, 6, 7, 8, 15], "latinhypercubesampl": [1, 4, 5, 7, 8, 13, 15], "unnormalis": [1, 4, 5, 7, 8, 13, 15], "gpytorch": [1, 2, 4, 5, 7, 8, 9, 10, 14, 15], "likelihood": [1, 2, 4, 5, 7, 8, 10, 14, 15], "gaussianlikelihood": [1, 4, 5, 7, 8, 15], "mll": [1, 4, 5, 7, 8, 10, 15], "exactmarginalloglikelihood": [1, 4, 5, 7, 8, 15], "pyro": 1, "infer": [1, 2, 14], "mcmc": 1, "nut": 1, "constraint": [1, 2, 4, 11, 14], "posit": 1, "prior": [1, 2], "uniformprior": 1, "set": [1, 2, 5, 14], "fast_comput": 1, "test": [1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15], "function": [1, 3, 4, 6, 7, 8, 10, 11, 14, 15], "func": [1, 4, 5, 7, 8, 11, 15], "minimis": [1, 4, 5, 7, 8, 11, 12, 15], "fals": [1, 4, 5, 7, 8, 9, 12, 15], "dim": [1, 4, 5, 7, 8, 9, 12, 13, 15], "bound": [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15], "train": [1, 2, 4, 5, 7, 8, 9, 10, 14, 15], "data": [1, 2, 4, 5, 7, 8, 9, 14, 15], "manual_se": [1, 4, 5, 7, 8, 15], "lh": [1, 4, 5, 7, 8, 15], "x_train": [1, 4, 5, 7, 8, 10, 15], "maximin": [1, 4, 5, 7, 8, 13, 14, 15], "point": [1, 2, 4, 5, 6, 9, 10, 12, 13, 14], "5": [1, 2, 4, 5, 7, 8, 10, 14, 15], "y_train": [1, 4, 5, 7, 8, 10, 15], "loop": [1, 2, 4, 5, 7, 8, 11, 14, 15], "iter": [1, 2, 4, 5, 7, 8, 14, 15], "40": [1, 4, 5, 15], "rang": [1, 2, 4, 5, 7, 8, 14, 15], "specifi": [1, 2, 4, 5, 7, 8, 14, 15], "noise_constraint": 1, "gp": [1, 2, 4, 5, 7, 8, 9, 10, 15], "mean_modul": [1, 10], "register_prior": 1, "mean_prior": 1, "constant": [1, 5, 10], "covar_modul": [1, 10], "base_kernel": 1, "lengthscale_prior": 1, "0": [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], "01": [1, 7, 8], "lengthscal": 1, "outputscale_prior": 1, "2": [1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15], "outputscal": 1, "noise_prior": 1, "nois": [1, 2, 12], "up": [1, 5], "sampl": [1, 2, 7, 8, 9, 11, 14], "def": 1, "pyro_gp": 1, "x": [1, 2, 4, 9, 10, 11, 12, 13], "y": [1, 2, 10, 13], "sampled_gp": 1, "pyro_sample_from_prior": 1, "output": [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 15], "ob": 1, "return": [1, 2, 9, 10, 11, 13, 14], "run": [1, 2, 5, 14], "nuts_kernel": 1, "mcmc_run": 1, "num_sampl": [1, 11], "128": 1, "warmup_step": 1, "disable_progbar": 1, "true": [1, 2, 5, 9, 12], "load": 1, "pyro_load_from_sampl": 1, "get_sampl": 1, "acq": [1, 4, 5, 7, 8, 15], "y_best": [1, 4, 7, 8, 9, 15], "max": [1, 2, 4, 5, 7, 8, 15], "beta": [1, 2, 4, 5, 7, 8, 9, 15], "96": [1, 4, 5, 7, 8, 15], "x_new": [1, 4, 5, 7, 8, 15], "_": [1, 2, 4, 5, 7, 8, 15], "lambda": [1, 4], "sum": 1, "num_start": [1, 4, 5, 7, 8, 11, 15], "evalu": [1, 2, 3, 4, 5, 6, 7, 8, 14, 15], "new": [1, 2, 4, 5, 7, 8, 14, 15], "y_new": [1, 4, 5, 7, 8, 15], "add": [1, 2, 4, 5, 7, 8, 15], "vstack": [1, 4, 5, 7, 8, 15], "hstack": [1, 4, 5, 7, 8, 15], "print": [1, 4, 5, 7, 8, 15], "best": [1, 2, 4, 5, 7, 8, 9, 11, 14, 15], "f": [1, 2, 4, 5, 7, 8, 15], "len": [1, 4, 5, 15], "t": [1, 2, 4, 5, 7, 8, 15], "input": [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15], "numpi": [1, 4, 5, 7, 8, 13, 15], "reshap": [1, 4, 5, 7, 8, 15], "round": [1, 4, 5, 7, 8, 15], "4": [1, 2, 4, 5, 7, 8, 9, 12, 14, 15], "result": [1, 2, 4, 5, 7, 8, 11, 14, 15], "best_it": [1, 4, 5, 7, 8, 15], "int": [1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], "argmax": [1, 4, 5, 7, 8, 15], "solut": [1, 2, 4, 5, 7, 8, 14, 15], "float": [1, 4, 5, 7, 8, 9, 10, 11, 12, 15], "4f": [1, 4, 5, 7, 8, 15], "33": [1, 5, 7, 15], "4456": 1, "1017": 1, "6031": 1, "9557": 1, "447": 1, "0752": 1, "351": 1, "262": 1, "3207": 1, "7352": 1, "5521": 1, "45": [1, 7], "4422": 1, "9751": 1, "7018": 1, "508": 1, "0311": 1, "7326": 1, "49": [1, 5, 15], "1384": 1, "1482": 1, "4061": 1, "2193": 1, "2555": 1, "7053": 1, "9284": 1, "59": 1, "1107": 1, "2156": 1, "4114": 1, "3149": 1, "297": 1, "7104": 1, "3": [1, 2, 4, 5, 6, 7, 8, 12, 14, 15], "0154": 1, "62": [1, 5, 15], "4298": 1, "8893": 1, "7379": 1, "6256": 1, "166": 1, "0193": 1, "66": [1, 4, 7], "1836": 1, "1595": 1, "434": 1, "2931": 1, "3465": 1, "6633": 1, "2275": 1, "aim": [2, 14], "solv": 2, "d": [2, 3, 9, 10, 11, 12, 13], "dimension": [2, 5, 12], "boldsymbol": 2, "arg": 2, "max_": 2, "mathcal": 2, "where": [2, 6, 14], "space": [2, 11, 12, 13, 14], "i": [2, 5, 6, 9, 10, 11, 14], "usual": [2, 14], "continu": [2, 14], "hyper": [2, 5, 14], "rectangl": 2, "b": [2, 5, 9, 11, 12, 14], "mathbb": 2, "r": [2, 14], "The": [2, 5, 6, 14], "most": 2, "commonli": 2, "deriv": 2, "free": 2, "expens": [2, 3, 6, 14], "black": [2, 3, 6, 14], "box": [2, 3, 6, 14], "onli": 2, "allow": [2, 14], "x_i": 2, "querri": 2, "y_i": 2, "observ": 2, "without": 2, "gain": [2, 14], "ani": [2, 10, 11], "further": 2, "insight": 2, "underli": 2, "system": [2, 14], "we": [2, 5], "assum": 2, "epsilon": 2, "introduc": 2, "when": [2, 6], "take": 2, "measur": 2, "independ": 2, "ident": 2, "distribut": [2, 5, 6, 10, 14], "gaussian": [2, 4, 5, 6, 7, 8, 9, 12, 14, 15], "sim": 2, "n": [2, 9, 10, 12, 13], "sigma": 2, "henc": 2, "pair": 2, "correspond": [2, 5], "defin": [2, 4, 5, 14], "d_n": 2, "matrix": [2, 10, 14], "x_n": 2, "vector": [2, 10], "y_n": 2, "base": [2, 9, 10, 12, 13, 14], "algorithm": [2, 5, 6, 10, 11, 14], "object": [2, 6, 12, 13, 14], "minimum": [2, 14], "number": [2, 9, 11, 12, 13, 14], "doe": [2, 14], "have": [2, 9, 14], "known": [2, 14], "mathemat": [2, 3, 14], "express": [2, 14], "everi": [2, 14], "requir": [2, 14], "cost": [2, 14], "effect": [2, 14], "effici": [2, 14], "routin": [2, 14], "meet": [2, 14], "criteria": [2, 14], "repres": [2, 14], "through": [2, 14], "m": [2, 14], "often": [2, 14], "process": [2, 4, 5, 6, 7, 8, 9, 14, 15], "thi": [2, 5, 6, 14], "represent": [2, 14], "can": [2, 5, 6, 9, 14], "us": [2, 5, 6, 9, 10, 11, 14], "find": [2, 14], "next": [2, 14], "should": [2, 6, 9, 14], "criterion": [2, 14], "an": [2, 5, 6, 14], "alpha": 2, "A": [2, 12, 14], "popular": [2, 14], "exampl": [2, 14], "expect": [2, 9, 14], "improv": [2, 9, 14], "better": [2, 14], "than": [2, 14], "previou": [2, 14], "perform": [2, 14], "_n": 2, "fit": [2, 4, 5, 7, 8, 10, 14, 15], "befor": [2, 14], "suggest": [2, 14], "ad": [2, 14], "itself": [2, 14], "restart": [2, 5, 14], "more": [2, 14], "inform": [2, 6, 14], "about": [2, 6, 14], "each": [2, 14], "mani": [2, 14], "budget": [2, 5, 14], "until": [2, 14], "satisfi": [2, 14], "found": [2, 14], "unitl": [2, 14], "pre": [2, 5, 14], "stop": [2, 14], "met": [2, 14], "initi": [2, 5, 14], "n_0": 2, "x_0": 2, "via": [2, 5, 14], "fill": 2, "design": [2, 14], "gather": 2, "y_0": 2, "while": [2, 14], "leq": 2, "do": 2, "comput": [2, 6, 9, 10, 12, 14], "increment": 2, "end": 2, "highest": 2, "choic": 2, "act": 2, "flexibl": 2, "non": 2, "parametr": 2, "regress": 2, "finit": 2, "collect": 2, "random": [2, 13, 14], "variabl": 2, "ha": 2, "joint": [2, 6, 11], "mean": [2, 5, 10, 13], "mu_0": 2, "mapsto": 2, "covari": [2, 10], "kernel": [2, 5, 10], "sigma_0": 2, "time": 2, "k": 2, "size": [2, 7, 8, 9, 10, 11, 12, 13], "over": 2, "all": [2, 5, 6, 9, 12, 14], "between": [2, 13], "posterior": 2, "predict": 2, "n_": 2, "x_": 2, "multivari": [2, 10], "normal": [2, 3, 10], "condit": 2, "some": [2, 5, 6, 9, 10, 12], "mid": 2, "left": 2, "mu_n": 2, "2_n": 2, "right": 2, "paramet": [2, 5, 9, 11, 12, 13, 14], "theta": 2, "varianc": 2, "estim": [2, 5, 14], "maximum": [2, 5, 10, 12, 14], "mle": [2, 5, 10, 14], "posteriori": [2, 14], "map": [2, 14], "fulli": [2, 6, 14], "packag": [2, 3, 14], "veri": 2, "power": [2, 14], "implement": [2, 5, 6, 11, 14], "wide": [2, 14], "select": [2, 13, 14], "exact": [2, 14], "approxim": [2, 14], "even": [2, 14], "deep": [2, 14], "It": [2, 6, 14], "also": [2, 6, 14], "come": 2, "rich": 2, "document": [2, 6, 14], "practic": 2, "larg": [2, 13], "commun": 2, "help": 2, "need": 2, "assess": 2, "good": [2, 6], "potenti": 2, "thu": 2, "current": [2, 9], "being": [2, 14], "global": [2, 12, 14], "optimum": [2, 5, 12], "To": 2, "balanc": 2, "explor": 2, "exploit": 2, "former": 2, "characteris": 2, "area": 2, "lack": 2, "uncertainti": 2, "high": 2, "latter": 2, "promis": 2, "trade": [2, 5, 9], "off": [2, 5, 9], "ensur": 2, "converg": 2, "first": [2, 5, 6], "local": 2, "full": 2, "support": [2, 14], "two": 2, "ar": [2, 6, 14], "ground": 2, "histori": 2, "theoret": 2, "empir": 2, "research": [2, 6, 14], "ei": 2, "biggest": 2, "upper": [2, 5, 9, 14], "confid": [2, 5, 9, 14], "ucb": 2, "optimist": 2, "view": 2, "user": 2, "level": 2, "alpha_": 2, "phi": 2, "z": 2, "sigma_n": 2, "frac": 2, "cdot": 2, "standard": [2, 12, 13], "deviat": [2, 12, 13], "cumul": 2, "probabl": 2, "densiti": 2, "sqrt": 2, "both": 2, "analyt": [2, 5, 14], "them": [2, 14], "determinist": [2, 9, 14], "l": [2, 5, 9, 11, 14], "bfg": [2, 5, 9, 11, 14], "unconstraint": 2, "slsqp": [2, 4, 9, 11, 14], "howev": 2, "sequenti": [2, 6, 11, 14], "singl": [2, 6, 14], "case": [2, 5], "which": [2, 5, 11], "immediatlei": 2, "repeat": 2, "For": [2, 5], "parallel": [2, 6, 14], "multi": [2, 5, 6, 14], "batch": [2, 11, 14], "asynchron": [2, 6, 14], "gener": [2, 5, 6, 11, 14], "intract": 2, "mont": [2, 11, 14], "carlo": [2, 11, 14], "idea": 2, "draw": [2, 11, 13], "directli": [2, 5], "predicitv": 2, "averag": [2, 9], "method": [2, 6, 7, 8, 9, 10, 11, 12, 13, 14], "made": 2, "viabl": 2, "reparameteris": 2, "utilis": 2, "mc": 2, "relu": 2, "pi": 2, "lvert": 2, "rvert": 2, "lower": 2, "triangular": 2, "choleski": 2, "decomposit": 2, "rectifi": 2, "linear": 2, "unit": [2, 13], "zero": [2, 10], "valu": 2, "below": 2, "leav": 2, "rest": 2, "due": 2, "stochast": [2, 9, 14], "adam": [2, 5, 7, 8, 9, 10, 11, 14], "evid": 2, "fix": [2, 9], "individu": 2, "affect": 2, "neg": [2, 9], "would": 2, "could": 2, "bia": 2, "furthermor": 2, "strategi": 2, "possibl": 2, "default": [2, 9, 10, 11], "approach": [2, 5], "second": 2, "option": [2, 10, 11], "greedi": [2, 11], "one": [2, 13], "after": [2, 14], "other": [2, 14], "hold": 2, "show": 2, "similarli": 2, "smaller": 2, "larger": 2, "increas": 2, "complex": 2, "leverag": 2, "same": 2, "properti": 2, "pend": [2, 9], "yet": 2, "been": 2, "treat": 2, "In": [2, 5], "wai": 2, "thei": 2, "consid": 2, "gardner": [2, 14], "jacob": [2, 14], "geoff": [2, 14], "pleiss": [2, 14], "kilian": [2, 14], "q": [2, 14], "weinberg": [2, 14], "david": [2, 14], "bindel": [2, 14], "andrew": [2, 3, 14], "g": [2, 14], "wilson": [2, 3, 14], "blackbox": [2, 14], "gpu": [2, 14], "acceler": [2, 14], "advanc": [2, 14], "neural": [2, 14], "31": [2, 8, 14], "2018": [2, 14], "jone": [2, 14], "donald": [2, 14], "matthia": [2, 14], "schonlau": [2, 14], "william": [2, 14], "j": [2, 14], "welch": [2, 14], "optim": [2, 3, 9, 10, 11, 14], "journal": [2, 3, 14], "13": [2, 14], "1998": [2, 14], "455": [2, 14], "sriniva": [2, 14], "niranjan": [2, 14], "andrea": [2, 14], "kraus": [2, 14], "sham": [2, 14], "kakad": [2, 14], "seeger": [2, 14], "bandit": [2, 14], "No": [2, 14], "regret": [2, 14], "experiment": [2, 6, 14], "proceed": [2, 14], "27th": [2, 14], "intern": [2, 14], "confer": [2, 14], "machin": [2, 14], "learn": [2, 10, 11, 14], "2010": [2, 14], "1015": [2, 14], "1022": [2, 14], "jame": [2, 14], "frank": [2, 14], "hutter": [2, 14], "marc": [2, 14], "deisenroth": [2, 14], "maxim": [2, 14], "kingma": [2, 14], "diederik": [2, 14], "p": [2, 12, 14], "jimmi": [2, 14], "ba": [2, 14], "3rd": [2, 14], "2015": [2, 14], "pleas": 3, "cite": 3, "diessner": 3, "mike": 3, "kevin": 3, "richard": 3, "whallei": 3, "transpar": [3, 14], "python": [3, 14], "bayesian": [3, 5], "optimis": [3, 9, 10], "arxiv": 3, "preprint": 3, "1234": 3, "56789": 3, "2023": 3, "bibtex": 3, "articl": 3, "nubo2023": 3, "titl": 3, "author": 3, "year": 3, "o": 3, "connor": 3, "joseph": 3, "wynn": 3, "sylvain": 3, "laizet": 3, "analysi": 3, "streamwis": 3, "vari": 3, "wall": 3, "blow": 3, "turbul": 3, "boundari": 3, "layer": 3, "flow": 3, "comust": 3, "yu": 3, "guan": 3, "investig": 3, "applic": 3, "fluid": [3, 6, 14], "dynam": [3, 6, 14], "frontier": 3, "appli": [3, 14], "statist": 3, "2022": 3, "fit_gp": [4, 5, 7, 8, 10, 15], "lr": [4, 5, 7, 8, 10, 11, 15], "step": [4, 5, 7, 8, 9, 10, 11, 15], "200": [4, 5, 7, 8, 10, 11, 15], "con": 4, "type": 4, "ineq": 4, "fun": 4, "eq": 4, "2442": 4, "2426": 4, "2574": 4, "3801": 4, "3812": 4, "1807": 4, "6823": 4, "1706": 4, "43": [4, 8], "2159": 4, "2841": 4, "3293": 4, "3916": 4, "2908": 4, "5618": 4, "4406": 4, "46": 4, "1803": 4, "1897": 4, "3561": 4, "3714": 4, "2694": 4, "6034": 4, "8027": 4, "52": 4, "2436": 4, "1538": 4, "4723": 4, "3641": 4, "2724": 4, "6077": 4, "9484": 4, "56": 4, "2045": 4, "1369": 4, "5032": 4, "3406": 4, "2733": 4, "6303": 4, "0979": 4, "57": [4, 5, 7, 15], "208": 4, "1383": 4, "457": 4, "257": 4, "3181": 4, "6691": 4, "65": [4, 5, 15], "2012": 4, "1452": 4, "4682": 4, "2816": 4, "3081": 4, "6545": 4, "3196": 4, "1455": 4, "4683": 4, "281": 4, "3084": 4, "6548": 4, "32": 4, "67": [4, 8], "2009": 4, "1458": 4, "4686": 4, "2803": 4, "3089": 4, "655": 4, "3204": 4, "68": 4, "2008": 4, "1463": 4, "4695": 4, "2798": 4, "3092": 4, "6552": 4, "3208": 4, "its": [5, 6], "depend": 5, "github": [5, 6], "repositori": 5, "pip": 5, "follow": 5, "code": [5, 6], "virtual": 5, "environ": 5, "recommend": 5, "git": 5, "http": 5, "com": 5, "mikediessn": 5, "want": [5, 6], "choos": 5, "6": [5, 7, 12], "hartmann": 5, "modal": 5, "Then": 5, "decid": 5, "per": 5, "dimens": [5, 9, 12, 13], "total": [5, 11], "30": 5, "now": 5, "prepar": 5, "matern": [5, 10], "95": 5, "interv": 5, "five": 5, "give": [5, 6], "70": 5, "3159": [5, 15], "8291": [5, 15], "5973": [5, 15], "3698": [5, 15], "39": [5, 15], "3975": [5, 15], "8033": [5, 15], "2752": [5, 15], "5622": [5, 15], "635": [5, 15], "8438": [5, 15], "42": [5, 15], "4225": [5, 15], "8332": [5, 15], "5555": [5, 15], "0419": [5, 15], "4017": [5, 15], "8468": [5, 15], "5813": [5, 15], "0085": [5, 8, 15], "1208": [5, 15], "51": [5, 15], "4026": [5, 15], "8657": [5, 15], "5994": [5, 15], "0246": [5, 15], "1584": [5, 15], "55": [5, 15], "4253": [5, 15], "8895": [5, 15], "5895": [5, 15], "0399": [5, 15], "1627": [5, 15], "3971": [5, 15], "9002": [5, 15], "5686": [5, 15], "0405": [5, 15], "1823": [5, 15], "4015": [5, 15], "873": [5, 15], "5623": [5, 15], "0426": [5, 15], "1861": [5, 15], "63": [5, 15], "4035": [5, 15], "8879": [5, 15], "5812": [5, 15], "0375": [5, 15], "1921": [5, 15], "64": [5, 15], "4045": [5, 15], "8868": [5, 15], "5793": [5, 15], "038": [5, 15], "1932": [5, 15], "4071": [5, 15], "8853": [5, 15], "5749": [5, 15], "0374": [5, 15], "1939": [5, 15], "final": 5, "overal": 5, "approximati": 5, "3224": 5, "short": [6, 14], "newcastl": [6, 14], "univers": [6, 14], "framework": [6, 14], "physic": [6, 14], "experi": [6, 14], "simul": [6, 14], "develop": [6, 14], "group": [6, 14], "focus": [6, 14], "precis": [6, 14], "make": [6, 14], "access": [6, 14], "disciplin": [6, 14], "written": [6, 14], "under": [6, 14], "bsd": [6, 14], "claus": [6, 14], "licens": [6, 14], "section": 6, "contain": [6, 12], "depth": 6, "explan": 6, "compon": 6, "surrog": [6, 14], "quickstart": 6, "guid": 6, "so": 6, "you": 6, "start": [6, 11, 14], "your": 6, "minut": 6, "place": 6, "journei": 6, "overview": 6, "get": 6, "primer": 6, "citat": 6, "provid": [6, 13, 14], "problem": [6, 12, 14], "capabl": 6, "boilerpl": 6, "tailor": 6, "specfic": 6, "constrain": 6, "detail": 6, "": 6, "go": 6, "sure": 6, "how": 6, "specif": 6, "modul": 6, "mcexpectedimprov": [7, 8, 9], "mcupperconfidencebound": [7, 8, 9], "10": [7, 8, 11], "256": [7, 8], "batch_siz": [7, 8, 11], "best_ev": [7, 8], "4206": 7, "9388": 7, "7492": 7, "4295": 7, "4485": 7, "0024": 7, "4497": 7, "38": 7, "4075": 7, "998": 7, "0014": 7, "5362": 7, "0043": 7, "0011": 7, "676": 7, "030e": 7, "8": [7, 8], "936e": 7, "9": [7, 8], "995e": 7, "419e": 7, "400e": 7, "03": 7, "000e": [7, 8], "04": [7, 8], "0968": 7, "47": [7, 8], "081e": 7, "699e": 7, "953e": 7, "815e": 7, "1251": 7, "61": 7, "980e": 7, "873e": 7, "979e": 7, "819e": 7, "7": 7, "1261": 7, "4037": 7, "8744": 7, "5707": 7, "0075": 7, "0469": 7, "1792": 7, "221e": 8, "083e": 8, "117e": 8, "429e": 8, "282e": 8, "6071": 8, "35": 8, "4154": 8, "9929": 8, "992": 8, "5121": 8, "0048": 8, "0012": 8, "7314": 8, "4165": 8, "915": 8, "9966": 8, "4999": 8, "073": 8, "9413": 8, "4151": 8, "8767": 8, "9961": 8, "5564": 8, "0177": 8, "1591": 8, "4031": 8, "8638": 8, "9958": 8, "5584": 8, "001": 8, "0313": 8, "1753": 8, "acquisition_funct": 9, "acquisitionfunct": 9, "sourc": [9, 10, 11, 12, 13], "tensor": [9, 10, 11, 12, 13], "attribut": [9, 10, 12, 13], "eval": [9, 12], "imrpov": 9, "none": [9, 10, 11, 12, 13], "monte_carlo": 9, "x_pend": 9, "512": 9, "fix_base_sampl": 9, "bool": [9, 12], "whether": 9, "If": 9, "base_sampl": 9, "nonetyp": 9, "drawn": 9, "class": [10, 13], "gaussian_process": 10, "exactgp": 10, "automat": 10, "relev": 10, "determin": 10, "forward": 10, "multivariatenorm": 10, "predictic": 10, "marginalloglikelihood": 10, "kwarg": [10, 11], "target": 10, "margin": 10, "log": 10, "rate": [10, 11], "keyword": [10, 11], "argument": [10, 11], "pass": [10, 11], "callabl": 11, "100": 11, "tupl": 11, "scipi": 11, "minim": [11, 13], "pick": 11, "latin": [11, 14], "hypercub": [11, 14], "initialis": 11, "optims": 11, "best_result": 11, "best_func_result": 11, "dict": [11, 12], "list": 11, "pytorch": 11, "multipoint": 11, "str": 11, "One": 11, "minimz": 11, "batch_result": 11, "sizq": 11, "batch_func_result": 11, "gen_candid": 11, "num_candid": 11, "candid": 11, "testfunct": 12, "noise_std": 12, "maximis": [12, 14], "c": 12, "dixonpric": 12, "hartmann3d": 12, "sumsquar": 12, "latin_hypercub": 13, "1000": 13, "largest": 13, "distanc": 13, "ndarrai": 13, "transfrom": 13, "standardis": 13, "subtract": 13, "divid": 13, "normalis": 13, "cube": 13, "rever": 13, "scale": 13, "refer": 14, "optimnis": 14, "still": 14, "restrict": 14, "synthet": 14, "ten": 14, "valid": 14}, "objects": {"nubo.acquisition": [[9, 0, 0, "-", "acquisition_function"], [9, 0, 0, "-", "analytical"], [9, 0, 0, "-", "monte_carlo"]], "nubo.acquisition.acquisition_function": [[9, 1, 1, "", "AcquisitionFunction"]], "nubo.acquisition.analytical": [[9, 1, 1, "", "ExpectedImprovement"], [9, 1, 1, "", "UpperConfidenceBound"]], "nubo.acquisition.analytical.ExpectedImprovement": [[9, 2, 1, "", "eval"]], "nubo.acquisition.analytical.UpperConfidenceBound": [[9, 2, 1, "", "eval"]], "nubo.acquisition.monte_carlo": [[9, 1, 1, "", "MCExpectedImprovement"], [9, 1, 1, "", "MCUpperConfidenceBound"]], "nubo.acquisition.monte_carlo.MCExpectedImprovement": [[9, 2, 1, "", "eval"]], "nubo.acquisition.monte_carlo.MCUpperConfidenceBound": [[9, 2, 1, "", "eval"]], "nubo.models": [[10, 0, 0, "-", "fit"], [10, 0, 0, "-", "gaussian_process"]], "nubo.models.fit": [[10, 3, 1, "", "fit_gp"]], "nubo.models.gaussian_process": [[10, 1, 1, "", "GaussianProcess"]], "nubo.models.gaussian_process.GaussianProcess": [[10, 2, 1, "", "forward"]], "nubo.optimisation": [[11, 0, 0, "-", "adam"], [11, 0, 0, "-", "lbfgsb"], [11, 0, 0, "-", "multipoint"], [11, 0, 0, "-", "slsqp"], [11, 0, 0, "-", "utils"]], "nubo.optimisation.adam": [[11, 3, 1, "", "adam"]], "nubo.optimisation.lbfgsb": [[11, 3, 1, "", "lbfgsb"]], "nubo.optimisation.multipoint": [[11, 3, 1, "", "joint"], [11, 3, 1, "", "sequential"]], "nubo.optimisation.slsqp": [[11, 3, 1, "", "slsqp"]], "nubo.optimisation.utils": [[11, 3, 1, "", "gen_candidates"]], "nubo.test_functions": [[12, 0, 0, "-", "ackley"], [12, 0, 0, "-", "dixonprice"], [12, 0, 0, "-", "griewank"], [12, 0, 0, "-", "hartmann"], [12, 0, 0, "-", "levy"], [12, 0, 0, "-", "rastrigin"], [12, 0, 0, "-", "schwefel"], [12, 0, 0, "-", "sphere"], [12, 0, 0, "-", "sumsquares"], [12, 0, 0, "-", "test_functions"], [12, 0, 0, "-", "zakharov"]], "nubo.test_functions.ackley": [[12, 1, 1, "", "Ackley"]], "nubo.test_functions.ackley.Ackley": [[12, 2, 1, "", "eval"]], "nubo.test_functions.dixonprice": [[12, 1, 1, "", "DixonPrice"]], "nubo.test_functions.dixonprice.DixonPrice": [[12, 2, 1, "", "eval"]], "nubo.test_functions.griewank": [[12, 1, 1, "", "Griewank"]], "nubo.test_functions.griewank.Griewank": [[12, 2, 1, "", "eval"]], "nubo.test_functions.hartmann": [[12, 1, 1, "", "Hartmann3D"], [12, 1, 1, "", "Hartmann6D"]], "nubo.test_functions.hartmann.Hartmann3D": [[12, 2, 1, "", "eval"]], "nubo.test_functions.hartmann.Hartmann6D": [[12, 2, 1, "", "eval"]], "nubo.test_functions.levy": [[12, 1, 1, "", "Levy"]], "nubo.test_functions.levy.Levy": [[12, 2, 1, "", "eval"]], "nubo.test_functions.rastrigin": [[12, 1, 1, "", "Rastrigin"]], "nubo.test_functions.rastrigin.Rastrigin": [[12, 2, 1, "", "eval"]], "nubo.test_functions.schwefel": [[12, 1, 1, "", "Schwefel"]], "nubo.test_functions.schwefel.Schwefel": [[12, 2, 1, "", "eval"]], "nubo.test_functions.sphere": [[12, 1, 1, "", "Sphere"]], "nubo.test_functions.sphere.Sphere": [[12, 2, 1, "", "eval"]], "nubo.test_functions.sumsquares": [[12, 1, 1, "", "SumSquares"]], "nubo.test_functions.sumsquares.SumSquares": [[12, 2, 1, "", "eval"]], "nubo.test_functions.test_functions": [[12, 1, 1, "", "TestFunction"]], "nubo.test_functions.zakharov": [[12, 1, 1, "", "Zakharov"]], "nubo.test_functions.zakharov.Zakharov": [[12, 2, 1, "", "eval"]], "nubo.utils": [[13, 0, 0, "-", "latin_hypercube"], [13, 0, 0, "-", "transfrom"]], "nubo.utils.latin_hypercube": [[13, 1, 1, "", "LatinHypercubeSampling"]], "nubo.utils.latin_hypercube.LatinHypercubeSampling": [[13, 2, 1, "", "maximin"], [13, 2, 1, "", "random"]], "nubo.utils.transfrom": [[13, 3, 1, "", "normalise"], [13, 3, 1, "", "standardise"], [13, 3, 1, "", "unnormalise"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"asynchron": 0, "bayesian": [0, 1, 2, 4, 6, 7, 8, 14, 15], "optimis": [0, 1, 2, 4, 5, 6, 7, 8, 11, 14, 15], "fulli": 1, "gaussian": [1, 10], "process": [1, 10], "primer": 2, "maximis": 2, "problem": 2, "surrog": [2, 10], "model": 2, "acquisit": [2, 9], "function": [2, 5, 9, 12], "citat": 3, "select": 3, "public": 3, "us": 3, "nubo": [3, 5, 6], "constrain": 4, "get": 5, "start": 5, "instal": 5, "toi": 5, "transpar": 6, "python": 6, "packag": 6, "exampl": 6, "refer": 6, "parallel": [7, 8], "multi": [7, 8, 11], "point": [7, 8, 11, 15], "joint": 7, "sequenti": [8, 15], "modul": [9, 10, 11, 12, 13], "parent": [9, 12], "class": [9, 12], "analyt": 9, "mont": 9, "carlo": 9, "aquisit": 9, "hyper": 10, "paramet": 10, "estim": 10, "determinist": 11, "stochast": 11, "util": [11, 13], "test": 12, "acklei": 12, "dixon": 12, "price": 12, "griewank": 12, "hartmann": 12, "levi": 12, "rastrigin": 12, "schwefel": 12, "sphere": 12, "sum": 12, "squar": 12, "zakharov": 12, "latin": 13, "hypercub": 13, "sampl": 13, "data": 13, "transform": 13, "overview": 14, "content": 14, "singl": 15}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Asynchronous Bayesian Optimisation": [[0, "Asynchronous-Bayesian-Optimisation"]], "Fully Bayesian Gaussian Process for Bayesian Optimisation": [[1, "Fully-Bayesian-Gaussian-Process-for-Bayesian-Optimisation"]], "Primer on Bayesian optimisation": [[2, "primer-on-bayesian-optimisation"]], "Maximisation problem": [[2, "maximisation-problem"]], "Bayesian optimisation": [[2, "bayesian-optimisation"], [14, "bayesian-optimisation"]], "Surrogate model": [[2, "surrogate-model"]], "Acquisition function": [[2, "acquisition-function"]], "Citation": [[3, "citation"]], "Selected publications using NUBO": [[3, "selected-publications-using-nubo"]], "Constrained Bayesian Optimisation": [[4, "Constrained-Bayesian-Optimisation"]], "Get started": [[5, "Get-started"]], "Installing NUBO": [[5, "Installing-NUBO"]], "Optimising a toy function with NUBO": [[5, "Optimising-a-toy-function-with-NUBO"]], "NUBO: a transparent python package for Bayesian optimisation": [[6, "nubo-a-transparent-python-package-for-bayesian-optimisation"]], "NUBO": [[6, "nubo"]], "NUBO:": [[6, null]], "Examples": [[6, "examples"]], "Examples:": [[6, null]], "Package reference": [[6, "package-reference"]], "Package reference:": [[6, null]], "Parallel multi-point joint Bayesian Optimisation": [[7, "Parallel-multi-point-joint-Bayesian-Optimisation"]], "Parallel multi-point sequential Bayesian Optimisation": [[8, "Parallel-multi-point-sequential-Bayesian-Optimisation"]], "Acquisition module": [[9, "acquisition-module"]], "Parent class": [[9, "module-nubo.acquisition.acquisition_function"], [12, "module-nubo.test_functions.test_functions"]], "Analytical acquisition functions": [[9, "module-nubo.acquisition.analytical"]], "Monte Carlo aquisition functions": [[9, "module-nubo.acquisition.monte_carlo"]], "Surrogates module": [[10, "surrogates-module"]], "Gaussian Process": [[10, "module-nubo.models.gaussian_process"]], "Hyper-parameter estimation": [[10, "module-nubo.models.fit"]], "Optimisation module": [[11, "optimisation-module"]], "Deterministic optimisers": [[11, "module-nubo.optimisation.lbfgsb"]], "Stochastic optimisers": [[11, "module-nubo.optimisation.adam"]], "Multi-point optimisation": [[11, "module-nubo.optimisation.multipoint"]], "Optimisation utilities": [[11, "module-nubo.optimisation.utils"]], "Test function module": [[12, "test-function-module"]], "Ackley function": [[12, "module-nubo.test_functions.ackley"]], "Dixon-Price function": [[12, "module-nubo.test_functions.dixonprice"]], "Griewank function": [[12, "module-nubo.test_functions.griewank"]], "Hartmann function": [[12, "module-nubo.test_functions.hartmann"]], "Levy function": [[12, "module-nubo.test_functions.levy"]], "Rastrigin function": [[12, "module-nubo.test_functions.rastrigin"]], "Schwefel function": [[12, "module-nubo.test_functions.schwefel"]], "Sphere function": [[12, "module-nubo.test_functions.sphere"]], "Sum-of-Squares function": [[12, "module-nubo.test_functions.sumsquares"]], "Zakharov function": [[12, "module-nubo.test_functions.zakharov"]], "Utility module": [[13, "utility-module"]], "Latin Hypercube Sampling": [[13, "module-nubo.utils.latin_hypercube"]], "Data transformations": [[13, "module-nubo.utils.transfrom"]], "Overview": [[14, "overview"]], "Contents": [[14, "contents"]], "Sequential single-point Bayesian Optimisation": [[15, "Sequential-single-point-Bayesian-Optimisation"]]}, "indexentries": {"acquisitionfunction (class in nubo.acquisition.acquisition_function)": [[9, "nubo.acquisition.acquisition_function.AcquisitionFunction"]], "expectedimprovement (class in nubo.acquisition.analytical)": [[9, "nubo.acquisition.analytical.ExpectedImprovement"]], "mcexpectedimprovement (class in nubo.acquisition.monte_carlo)": [[9, "nubo.acquisition.monte_carlo.MCExpectedImprovement"]], "mcupperconfidencebound (class in nubo.acquisition.monte_carlo)": [[9, "nubo.acquisition.monte_carlo.MCUpperConfidenceBound"]], "upperconfidencebound (class in nubo.acquisition.analytical)": [[9, "nubo.acquisition.analytical.UpperConfidenceBound"]], "eval() (nubo.acquisition.analytical.expectedimprovement method)": [[9, "nubo.acquisition.analytical.ExpectedImprovement.eval"]], "eval() (nubo.acquisition.analytical.upperconfidencebound method)": [[9, "nubo.acquisition.analytical.UpperConfidenceBound.eval"]], "eval() (nubo.acquisition.monte_carlo.mcexpectedimprovement method)": [[9, "nubo.acquisition.monte_carlo.MCExpectedImprovement.eval"]], "eval() (nubo.acquisition.monte_carlo.mcupperconfidencebound method)": [[9, "nubo.acquisition.monte_carlo.MCUpperConfidenceBound.eval"]], "module": [[9, "module-nubo.acquisition.acquisition_function"], [9, "module-nubo.acquisition.analytical"], [9, "module-nubo.acquisition.monte_carlo"], [10, "module-nubo.models.fit"], [10, "module-nubo.models.gaussian_process"], [11, "module-nubo.optimisation.adam"], [11, "module-nubo.optimisation.lbfgsb"], [11, "module-nubo.optimisation.multipoint"], [11, "module-nubo.optimisation.slsqp"], [11, "module-nubo.optimisation.utils"], [12, "module-nubo.test_functions.ackley"], [12, "module-nubo.test_functions.dixonprice"], [12, "module-nubo.test_functions.griewank"], [12, "module-nubo.test_functions.hartmann"], [12, "module-nubo.test_functions.levy"], [12, "module-nubo.test_functions.rastrigin"], [12, "module-nubo.test_functions.schwefel"], [12, "module-nubo.test_functions.sphere"], [12, "module-nubo.test_functions.sumsquares"], [12, "module-nubo.test_functions.test_functions"], [12, "module-nubo.test_functions.zakharov"], [13, "module-nubo.utils.latin_hypercube"], [13, "module-nubo.utils.transfrom"]], "nubo.acquisition.acquisition_function": [[9, "module-nubo.acquisition.acquisition_function"]], "nubo.acquisition.analytical": [[9, "module-nubo.acquisition.analytical"]], "nubo.acquisition.monte_carlo": [[9, "module-nubo.acquisition.monte_carlo"]], "gaussianprocess (class in nubo.models.gaussian_process)": [[10, "nubo.models.gaussian_process.GaussianProcess"]], "fit_gp() (in module nubo.models.fit)": [[10, "nubo.models.fit.fit_gp"]], "forward() (nubo.models.gaussian_process.gaussianprocess method)": [[10, "nubo.models.gaussian_process.GaussianProcess.forward"]], "nubo.models.fit": [[10, "module-nubo.models.fit"]], "nubo.models.gaussian_process": [[10, "module-nubo.models.gaussian_process"]], "adam() (in module nubo.optimisation.adam)": [[11, "nubo.optimisation.adam.adam"]], "gen_candidates() (in module nubo.optimisation.utils)": [[11, "nubo.optimisation.utils.gen_candidates"]], "joint() (in module nubo.optimisation.multipoint)": [[11, "nubo.optimisation.multipoint.joint"]], "lbfgsb() (in module nubo.optimisation.lbfgsb)": [[11, "nubo.optimisation.lbfgsb.lbfgsb"]], "nubo.optimisation.adam": [[11, "module-nubo.optimisation.adam"]], "nubo.optimisation.lbfgsb": [[11, "module-nubo.optimisation.lbfgsb"]], "nubo.optimisation.multipoint": [[11, "module-nubo.optimisation.multipoint"]], "nubo.optimisation.slsqp": [[11, "module-nubo.optimisation.slsqp"]], "nubo.optimisation.utils": [[11, "module-nubo.optimisation.utils"]], "sequential() (in module nubo.optimisation.multipoint)": [[11, "nubo.optimisation.multipoint.sequential"]], "slsqp() (in module nubo.optimisation.slsqp)": [[11, "nubo.optimisation.slsqp.slsqp"]], "ackley (class in nubo.test_functions.ackley)": [[12, "nubo.test_functions.ackley.Ackley"]], "dixonprice (class in nubo.test_functions.dixonprice)": [[12, "nubo.test_functions.dixonprice.DixonPrice"]], "griewank (class in nubo.test_functions.griewank)": [[12, "nubo.test_functions.griewank.Griewank"]], "hartmann3d (class in nubo.test_functions.hartmann)": [[12, "nubo.test_functions.hartmann.Hartmann3D"]], "hartmann6d (class in nubo.test_functions.hartmann)": [[12, "nubo.test_functions.hartmann.Hartmann6D"]], "levy (class in nubo.test_functions.levy)": [[12, "nubo.test_functions.levy.Levy"]], "rastrigin (class in nubo.test_functions.rastrigin)": [[12, "nubo.test_functions.rastrigin.Rastrigin"]], "schwefel (class in nubo.test_functions.schwefel)": [[12, "nubo.test_functions.schwefel.Schwefel"]], "sphere (class in nubo.test_functions.sphere)": [[12, "nubo.test_functions.sphere.Sphere"]], "sumsquares (class in nubo.test_functions.sumsquares)": [[12, "nubo.test_functions.sumsquares.SumSquares"]], "testfunction (class in nubo.test_functions.test_functions)": [[12, "nubo.test_functions.test_functions.TestFunction"]], "zakharov (class in nubo.test_functions.zakharov)": [[12, "nubo.test_functions.zakharov.Zakharov"]], "eval() (nubo.test_functions.ackley.ackley method)": [[12, "nubo.test_functions.ackley.Ackley.eval"]], "eval() (nubo.test_functions.dixonprice.dixonprice method)": [[12, "nubo.test_functions.dixonprice.DixonPrice.eval"]], "eval() (nubo.test_functions.griewank.griewank method)": [[12, "nubo.test_functions.griewank.Griewank.eval"]], "eval() (nubo.test_functions.hartmann.hartmann3d method)": [[12, "nubo.test_functions.hartmann.Hartmann3D.eval"]], "eval() (nubo.test_functions.hartmann.hartmann6d method)": [[12, "nubo.test_functions.hartmann.Hartmann6D.eval"]], "eval() (nubo.test_functions.levy.levy method)": [[12, "nubo.test_functions.levy.Levy.eval"]], "eval() (nubo.test_functions.rastrigin.rastrigin method)": [[12, "nubo.test_functions.rastrigin.Rastrigin.eval"]], "eval() (nubo.test_functions.schwefel.schwefel method)": [[12, "nubo.test_functions.schwefel.Schwefel.eval"]], "eval() (nubo.test_functions.sphere.sphere method)": [[12, "nubo.test_functions.sphere.Sphere.eval"]], "eval() (nubo.test_functions.sumsquares.sumsquares method)": [[12, "nubo.test_functions.sumsquares.SumSquares.eval"]], "eval() (nubo.test_functions.zakharov.zakharov method)": [[12, "nubo.test_functions.zakharov.Zakharov.eval"]], "nubo.test_functions.ackley": [[12, "module-nubo.test_functions.ackley"]], "nubo.test_functions.dixonprice": [[12, "module-nubo.test_functions.dixonprice"]], "nubo.test_functions.griewank": [[12, "module-nubo.test_functions.griewank"]], "nubo.test_functions.hartmann": [[12, "module-nubo.test_functions.hartmann"]], "nubo.test_functions.levy": [[12, "module-nubo.test_functions.levy"]], "nubo.test_functions.rastrigin": [[12, "module-nubo.test_functions.rastrigin"]], "nubo.test_functions.schwefel": [[12, "module-nubo.test_functions.schwefel"]], "nubo.test_functions.sphere": [[12, "module-nubo.test_functions.sphere"]], "nubo.test_functions.sumsquares": [[12, "module-nubo.test_functions.sumsquares"]], "nubo.test_functions.test_functions": [[12, "module-nubo.test_functions.test_functions"]], "nubo.test_functions.zakharov": [[12, "module-nubo.test_functions.zakharov"]], "latinhypercubesampling (class in nubo.utils.latin_hypercube)": [[13, "nubo.utils.latin_hypercube.LatinHypercubeSampling"]], "maximin() (nubo.utils.latin_hypercube.latinhypercubesampling method)": [[13, "nubo.utils.latin_hypercube.LatinHypercubeSampling.maximin"]], "normalise() (in module nubo.utils.transfrom)": [[13, "nubo.utils.transfrom.normalise"]], "nubo.utils.latin_hypercube": [[13, "module-nubo.utils.latin_hypercube"]], "nubo.utils.transfrom": [[13, "module-nubo.utils.transfrom"]], "random() (nubo.utils.latin_hypercube.latinhypercubesampling method)": [[13, "nubo.utils.latin_hypercube.LatinHypercubeSampling.random"]], "standardise() (in module nubo.utils.transfrom)": [[13, "nubo.utils.transfrom.standardise"]], "unnormalise() (in module nubo.utils.transfrom)": [[13, "nubo.utils.transfrom.unnormalise"]]}})