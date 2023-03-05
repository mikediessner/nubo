Search.setIndex({"docnames": ["acquisition_functions", "asynchronous_bo", "bayesian_gp", "bayesian_optimisation", "citation", "constrained_bo", "get_started", "index", "modules", "multipoint_joint", "multipoint_sequential", "nubo", "nubo.acquisition", "nubo.models", "nubo.optimisation", "nubo.test_functions", "nubo.utils", "overview", "singlepoint", "surrogate_models"], "filenames": ["acquisition_functions.md", "asynchronous_bo.ipynb", "bayesian_gp.ipynb", "bayesian_optimisation.md", "citation.md", "constrained_bo.ipynb", "get_started.ipynb", "index.rst", "modules.rst", "multipoint_joint.ipynb", "multipoint_sequential.ipynb", "nubo.rst", "nubo.acquisition.rst", "nubo.models.rst", "nubo.optimisation.rst", "nubo.test_functions.rst", "nubo.utils.rst", "overview.md", "singlepoint.ipynb", "surrogate_models.md"], "titles": ["Acquisition functions", "Asynchronous Bayesian Optimisation", "Fully Bayesian Gaussian Process for Bayesian Optimisation", "Bayesian Optimisation", "Citation", "Constrained Bayesian Optimisation", "Get started", "NUBO: a transparent python package for Bayesian Optimisation", "nubo", "Parallel multi-point joint Bayesian Optimisation", "Parallel multi-point sequential Bayesian Optimisation", "nubo package", "nubo.acquisition package", "nubo.models package", "nubo.optimisation package", "nubo.test_functions package", "nubo.utils package", "Overview", "Single-point Bayesian Optimisation", "Surrogate models"], "terms": {"1": [2, 5, 6, 9, 10, 12, 13, 14, 16, 18], "import": [2, 5, 6, 9, 10, 18], "torch": [2, 5, 6, 9, 10, 12, 18], "from": [2, 5, 6, 7, 9, 10, 18], "nubo": [2, 5, 9, 10, 18], "acquisit": [2, 5, 6, 7, 8, 9, 10, 11, 14, 18], "expectedimprov": [2, 5, 11, 12, 18], "upperconfidencebound": [2, 5, 6, 11, 12, 18], "model": [2, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18], "gaussianprocess": [2, 5, 6, 9, 10, 11, 13, 18], "lbfgsb": [2, 6, 8, 11, 18], "test_funct": [2, 5, 6, 7, 8, 9, 10, 11, 18], "hartmann6d": [2, 5, 6, 9, 10, 11, 15, 18], "util": [2, 5, 6, 7, 8, 9, 10, 11, 18], "latinhypercubesampl": [2, 5, 6, 9, 10, 11, 16, 18], "unnormalis": [2, 5, 6, 9, 10, 11, 16, 18], "gpytorch": [2, 5, 6, 9, 10, 12, 17, 18], "likelihood": [2, 5, 6, 9, 10, 13, 18], "gaussianlikelihood": [2, 5, 6, 9, 10, 18], "mll": [2, 5, 6, 9, 10, 13, 18], "exactmarginalloglikelihood": [2, 5, 6, 9, 10, 18], "pyro": 2, "infer": 2, "mcmc": 2, "nut": 2, "constraint": [2, 5, 14, 17], "posit": 2, "prior": 2, "uniformprior": 2, "set": [2, 6, 16, 17], "fast_comput": 2, "test": [2, 5, 6, 9, 10, 12, 17, 18], "function": [2, 4, 5, 7, 9, 10, 12, 13, 14, 18], "func": [2, 5, 6, 9, 10, 14, 18], "minimis": [2, 5, 6, 9, 10, 15, 18], "fals": [2, 5, 6, 9, 10, 12, 18], "dim": [2, 5, 6, 9, 10, 15, 16, 18], "bound": [2, 5, 6, 9, 10, 14, 16, 17, 18], "train": [2, 5, 6, 9, 10, 12, 18], "data": [2, 5, 6, 9, 10, 12, 16, 17, 18], "manual_se": [2, 5, 6, 9, 10, 18], "lh": [2, 5, 6, 9, 10, 18], "x_train": [2, 5, 6, 9, 10, 13, 18], "maximin": [2, 5, 6, 9, 10, 11, 16, 17, 18], "point": [2, 5, 6, 7, 12, 13, 16, 17], "5": [2, 5, 6, 9, 10, 13, 18], "y_train": [2, 5, 6, 9, 10, 13, 18], "loop": [2, 5, 6, 9, 10, 14, 18], "iter": [2, 5, 6, 9, 10, 18], "40": [2, 5, 6, 18], "rang": [2, 5, 6, 9, 10, 16, 18], "specifi": [2, 5, 6, 9, 10, 18], "noise_constraint": 2, "gp": [2, 5, 6, 9, 10, 12, 13, 18], "mean_modul": 2, "register_prior": 2, "mean_prior": 2, "constant": [2, 6, 13], "covar_modul": 2, "base_kernel": 2, "lengthscale_prior": 2, "0": [2, 5, 6, 9, 10, 13, 14, 15, 16, 18], "01": [2, 9, 10], "lengthscal": 2, "outputscale_prior": 2, "2": [2, 5, 6, 9, 10, 13, 16, 18], "outputscal": 2, "noise_prior": 2, "nois": 2, "up": [2, 6], "sampl": [2, 9, 10, 12, 16, 17], "def": 2, "pyro_gp": 2, "x": [2, 5, 12, 13, 16], "y": [2, 13, 16], "sampled_gp": 2, "pyro_sample_from_prior": 2, "output": [2, 5, 6, 9, 10, 12, 18], "ob": 2, "return": [2, 13, 16], "run": [2, 6], "nuts_kernel": 2, "mcmc_run": 2, "num_sampl": [2, 14], "128": 2, "warmup_step": 2, "disable_progbar": 2, "true": [2, 6, 15], "load": 2, "pyro_load_from_sampl": 2, "get_sampl": 2, "acq": [2, 5, 6, 9, 10, 18], "y_best": [2, 5, 9, 10, 12, 18], "max": [2, 5, 6, 9, 10, 18], "beta": [2, 5, 6, 9, 10, 12, 18], "96": [2, 5, 6, 9, 10, 18], "x_new": [2, 5, 6, 9, 10, 18], "_": [2, 5, 6, 9, 10, 18], "lambda": [2, 5], "sum": 2, "num_start": [2, 5, 6, 9, 10, 14, 18], "evalu": [2, 4, 5, 6, 7, 9, 10, 17, 18], "new": [2, 5, 6, 9, 10, 18], "y_new": [2, 5, 6, 9, 10, 18], "add": [2, 5, 6, 9, 10, 16, 18], "vstack": [2, 5, 6, 9, 10, 18], "hstack": [2, 5, 6, 9, 10, 18], "print": [2, 5, 6, 9, 10, 18], "best": [2, 5, 6, 9, 10, 12, 18], "f": [2, 5, 6, 9, 10, 17, 18], "len": [2, 5, 6, 18], "t": [2, 5, 6, 9, 10, 18], "input": [2, 5, 6, 9, 10, 18], "numpi": [2, 5, 6, 9, 10, 18], "reshap": [2, 5, 6, 9, 10, 18], "round": [2, 5, 6, 9, 10, 18], "4": [2, 5, 6, 9, 10, 17, 18], "result": [2, 5, 6, 9, 10, 18], "best_it": [2, 5, 6, 9, 10, 18], "int": [2, 5, 6, 9, 10, 12, 13, 14, 15, 16, 18], "argmax": [2, 5, 6, 9, 10, 18], "solut": [2, 5, 6, 9, 10, 18], "float": [2, 5, 6, 9, 10, 12, 13, 14, 15, 18], "4f": [2, 5, 6, 9, 10, 18], "33": [2, 6, 9, 18], "4456": 2, "1017": 2, "6031": 2, "9557": 2, "447": 2, "0752": 2, "351": 2, "262": 2, "3207": 2, "7352": 2, "5521": 2, "45": [2, 9], "4422": 2, "9751": 2, "7018": 2, "508": 2, "0311": 2, "7326": 2, "49": [2, 6, 18], "1384": 2, "1482": 2, "4061": 2, "2193": 2, "2555": 2, "7053": 2, "9284": 2, "59": 2, "1107": 2, "2156": 2, "4114": 2, "3149": 2, "297": 2, "7104": 2, "3": [2, 5, 6, 9, 10, 12, 18], "0154": 2, "62": [2, 6, 18], "4298": 2, "8893": 2, "7379": 2, "6256": 2, "166": 2, "0193": 2, "66": [2, 5, 9], "1836": 2, "1595": 2, "434": 2, "2931": 2, "3465": 2, "6633": 2, "2275": 2, "pleas": 4, "cite": 4, "diessner": 4, "mike": 4, "kevin": 4, "wilson": [4, 17], "richard": 4, "d": [4, 12, 16, 17], "whallei": 4, "transpar": 4, "python": 4, "packag": [4, 8], "bayesian": [4, 6, 17], "optimis": [4, 8, 11, 17], "arxiv": 4, "preprint": 4, "1234": 4, "56789": 4, "2023": 4, "bibtex": 4, "articl": 4, "nubo2023": 4, "titl": 4, "author": 4, "journal": [4, 17], "year": 4, "o": 4, "connor": 4, "joseph": 4, "andrew": 4, "wynn": 4, "sylvain": 4, "laizet": 4, "analysi": 4, "streamwis": 4, "vari": 4, "wall": 4, "normal": [4, 13], "blow": 4, "turbul": 4, "boundari": 4, "layer": 4, "flow": 4, "comust": 4, "yu": 4, "guan": 4, "investig": 4, "optim": [4, 17], "expens": [4, 7, 17], "black": [4, 7, 17], "box": [4, 7, 17], "applic": 4, "fluid": [4, 7, 17], "dynam": [4, 7, 17], "frontier": 4, "appli": 4, "mathemat": 4, "statist": 4, "2022": 4, "fit_gp": [5, 6, 9, 10, 11, 13, 18], "slsqp": [5, 8, 11, 17], "gaussian": [5, 6, 7, 9, 10, 12, 13, 17, 18], "process": [5, 6, 7, 9, 10, 12, 13, 17, 18], "fit": [5, 6, 8, 9, 10, 11, 18], "lr": [5, 6, 9, 10, 13, 14, 18], "step": [5, 6, 9, 10, 13, 14, 18], "200": [5, 6, 9, 10, 18], "defin": [5, 6, 16], "con": 5, "type": [5, 16], "ineq": 5, "fun": 5, "eq": 5, "2442": 5, "2426": 5, "2574": 5, "3801": 5, "3812": 5, "1807": 5, "6823": 5, "1706": 5, "43": [5, 10], "2159": 5, "2841": 5, "3293": 5, "3916": 5, "2908": 5, "5618": 5, "4406": 5, "46": 5, "1803": 5, "1897": 5, "3561": 5, "3714": 5, "2694": 5, "6034": 5, "8027": 5, "52": 5, "2436": 5, "1538": 5, "4723": 5, "3641": 5, "2724": 5, "6077": 5, "9484": 5, "56": 5, "2045": 5, "1369": 5, "5032": 5, "3406": 5, "2733": 5, "6303": 5, "0979": 5, "57": [5, 6, 9, 18], "208": 5, "1383": 5, "457": 5, "257": 5, "3181": 5, "6691": 5, "65": [5, 6, 18], "2012": 5, "1452": 5, "4682": 5, "2816": 5, "3081": 5, "6545": 5, "3196": 5, "1455": 5, "4683": 5, "281": 5, "3084": 5, "6548": 5, "32": 5, "67": [5, 10], "2009": 5, "1458": 5, "4686": 5, "2803": 5, "3089": 5, "655": 5, "3204": 5, "68": 5, "2008": 5, "1463": 5, "4695": 5, "2798": 5, "3092": 5, "6552": 5, "3208": 5, "all": [6, 7], "its": [6, 7], "depend": 6, "directli": 6, "github": [6, 7], "repositori": 6, "us": [6, 7, 17], "pip": 6, "follow": 6, "code": [6, 7], "The": [6, 7], "virtual": 6, "environ": 6, "i": [6, 7, 16], "recommend": 6, "git": 6, "http": 6, "com": 6, "mikediessn": 6, "first": [6, 16], "we": 6, "want": 6, "In": 6, "thi": [6, 7], "case": 6, "choos": 6, "6": [6, 9], "dimension": 6, "hartmann": [6, 8, 11], "multi": [6, 7, 14, 17], "modal": 6, "Then": 6, "gener": [6, 7, 14, 17], "some": [6, 7], "initi": [6, 17], "decid": 6, "per": 6, "dimens": [6, 16], "total": 6, "30": 6, "now": 6, "can": [6, 17], "prepar": 6, "pre": 6, "mean": [6, 13], "matern": [6, 13], "kernel": [6, 13], "estim": 6, "hyper": 6, "paramet": [6, 12, 16], "via": [6, 17], "maximum": 6, "mle": 6, "adam": [6, 8, 9, 10, 11, 17], "For": 6, "implement": [6, 7, 17], "analyt": [6, 12, 17], "upper": [6, 16, 17], "confid": [6, 17], "trade": 6, "off": 6, "correspond": 6, "95": 6, "interv": 6, "distribut": [6, 13], "l": [6, 14, 17], "bfg": [6, 14, 17], "b": [6, 14, 17], "algorithm": [6, 7, 17], "approach": 6, "five": 6, "restart": 6, "give": [6, 7], "an": [6, 7, 16], "budget": 6, "70": 6, "3159": [6, 18], "8291": [6, 18], "5973": [6, 18], "3698": [6, 18], "39": [6, 18], "3975": [6, 18], "8033": [6, 18], "2752": [6, 18], "5622": [6, 18], "635": [6, 18], "8438": [6, 18], "42": [6, 18], "4225": [6, 18], "8332": [6, 18], "5555": [6, 18], "0419": [6, 18], "4017": [6, 18], "8468": [6, 18], "5813": [6, 18], "0085": [6, 10, 18], "1208": [6, 18], "51": [6, 18], "4026": [6, 18], "8657": [6, 18], "5994": [6, 18], "0246": [6, 18], "1584": [6, 18], "55": [6, 18], "4253": [6, 18], "8895": [6, 18], "5895": [6, 18], "0399": [6, 18], "1627": [6, 18], "3971": [6, 18], "9002": [6, 18], "5686": [6, 18], "0405": [6, 18], "1823": [6, 18], "4015": [6, 18], "873": [6, 18], "5623": [6, 18], "0426": [6, 18], "1861": [6, 18], "63": [6, 18], "4035": [6, 18], "8879": [6, 18], "5812": [6, 18], "0375": [6, 18], "1921": [6, 18], "64": [6, 18], "4045": [6, 18], "8868": [6, 18], "5793": [6, 18], "038": [6, 18], "1932": [6, 18], "4071": [6, 18], "8853": [6, 18], "5749": [6, 18], "0374": [6, 18], "1939": [6, 18], "final": [6, 16], "overal": 6, "which": [6, 16], "approximati": 6, "optimum": 6, "3224": 6, "short": 7, "newcastl": [7, 17], "univers": [7, 17], "framework": [7, 17], "physic": 7, "experi": [7, 17], "comput": [7, 12, 13, 16], "simul": 7, "develop": [7, 17], "experiment": [7, 17], "research": [7, 17], "group": [7, 17], "It": 7, "focus": 7, "comprehens": 7, "provid": [7, 16], "extens": 7, "explan": 7, "method": [7, 9, 10, 14], "make": 7, "access": 7, "disciplin": 7, "section": 7, "contain": [7, 16], "inform": [7, 17], "about": 7, "depth": 7, "compon": 7, "surrog": [7, 17], "also": 7, "quickstart": 7, "guid": 7, "place": 7, "start": [7, 14, 17], "your": 7, "journei": 7, "overview": 7, "get": 7, "citat": 7, "problem": 7, "capabl": 7, "boilerpl": 7, "good": 7, "when": [7, 17], "specfic": 7, "singl": [7, 17], "parallel": [7, 17], "joint": [7, 11, 14, 17], "sequenti": [7, 11, 14, 17], "asynchron": [7, 17], "constrain": 7, "fulli": 7, "detail": 7, "document": 7, "": [7, 17], "go": 7, "you": 7, "ar": 7, "sure": 7, "how": 7, "specif": 7, "object": [7, 12, 15, 16], "should": [7, 16], "subpackag": 8, "submodul": [8, 11], "acquisition_funct": [8, 11], "modul": 8, "expected_improv": [8, 11], "upper_confidence_bound": [8, 11], "content": 8, "gaussian_process": [8, 11], "multipoint": [8, 11, 17], "acklei": [8, 11], "dixonpric": [8, 11], "griewank": [8, 11], "levi": [8, 11], "rastrigin": [8, 11], "schwefel": [8, 11], "sphere": [8, 11], "sumsquar": [8, 11], "zakharov": [8, 11], "latin_hypercub": [8, 11], "transfrom": [8, 11], "mcexpectedimprov": [9, 10, 11, 12], "mcupperconfidencebound": [9, 10, 11, 12], "10": [9, 10, 14], "256": [9, 10], "batch_siz": [9, 10, 14], "size": [9, 10, 12], "best_ev": [9, 10], "4206": 9, "9388": 9, "7492": 9, "4295": 9, "4485": 9, "0024": 9, "4497": 9, "38": 9, "4075": 9, "998": 9, "0014": 9, "5362": 9, "0043": 9, "0011": 9, "676": 9, "030e": 9, "8": [9, 10], "936e": 9, "9": [9, 10], "995e": 9, "419e": 9, "400e": 9, "03": 9, "000e": [9, 10], "04": [9, 10], "0968": 9, "47": [9, 10], "081e": 9, "699e": 9, "953e": 9, "815e": 9, "1251": 9, "61": 9, "980e": 9, "873e": 9, "979e": 9, "819e": 9, "7": 9, "1261": 9, "4037": 9, "8744": 9, "5707": 9, "0075": 9, "0469": 9, "1792": 9, "31": [10, 17], "221e": 10, "083e": 10, "117e": 10, "429e": 10, "282e": 10, "6071": 10, "35": 10, "4154": 10, "9929": 10, "992": 10, "5121": 10, "0048": 10, "0012": 10, "7314": 10, "4165": 10, "915": 10, "9966": 10, "4999": 10, "073": 10, "9413": 10, "4151": 10, "8767": 10, "9961": 10, "5564": 10, "0177": 10, "1591": 10, "4031": 10, "8638": 10, "9958": 10, "5584": 10, "001": 10, "0313": 10, "1753": 10, "acquisitionfunct": [11, 12], "eval": [11, 12], "forward": [11, 13], "gen_candid": [11, 14], "hartmann3d": [11, 15], "testfunct": [11, 15], "random": [11, 16], "normalis": [11, 16], "standardis": [11, 16], "class": [12, 13, 15, 16], "sourc": [12, 13, 14, 15, 16], "base": [12, 13, 15, 16, 17], "tensor": [12, 13, 14, 16], "expect": [12, 17], "improv": [12, 17], "neg": 12, "x_pend": 12, "none": [12, 13, 14, 15, 16], "512": 12, "fix_base_sampl": 12, "bool": [12, 15], "averag": 12, "mont": [12, 14, 17], "carlo": [12, 14, 17], "8415999999999997": 12, "marginalloglikelihood": 13, "100": [13, 14], "kwarg": [13, 14], "exactgp": 13, "multivariatenorm": 13, "covari": 13, "multivari": 13, "callabl": 14, "ani": 14, "tupl": 14, "str": 14, "dict": 14, "list": 14, "num_candid": 14, "candid": 14, "noise_std": 15, "1000": 16, "draw": 16, "latin": [16, 17], "hypercub": [16, 17], "larg": [16, 17], "number": 16, "minim": 16, "distanc": 16, "between": 16, "each": 16, "pick": 16, "largest": 16, "consid": 16, "maximin_hypercub": 16, "n": [16, 17], "arrai": 16, "where": 16, "ndarrai": 16, "permut": 16, "integ": 16, "quantil": 16, "valu": 16, "ha": 16, "belong": 16, "translat": 16, "individu": 16, "uniform": 16, "u": 16, "2xd": 16, "row": 16, "lower": 16, "second": 16, "rever": 16, "jone": 17, "et": 17, "al": 17, "1998": 17, "sriniva": 17, "2010": 17, "2018": 17, "determinist": 17, "stochast": 17, "design": 17, "synthet": 17, "one": 17, "ten": 17, "befor": 17, "deploi": 17, "properti": 17, "singlepoint": 17, "ye": 17, "No": 17, "strategi": 17, "small": 17, "batch": 17, "fix": 17, "pend": 17, "ad": 17, "r": 17, "schonlau": 17, "m": 17, "welch": 17, "w": 17, "j": 17, "effici": 17, "global": 17, "13": 17, "455": 17, "kraus": 17, "A": 17, "kakad": 17, "seeger": 17, "bandit": 17, "regret": 17, "proceed": 17, "27th": 17, "intern": 17, "confer": 17, "machin": 17, "learn": 17, "1015": 17, "1022": 17, "hutter": 17, "deisenroth": 17, "maxim": 17, "advanc": 17, "neural": 17, "system": 17}, "objects": {"": [[11, 0, 0, "-", "nubo"]], "nubo": [[12, 0, 0, "-", "acquisition"], [13, 0, 0, "-", "models"], [14, 0, 0, "-", "optimisation"], [15, 0, 0, "-", "test_functions"], [16, 0, 0, "-", "utils"]], "nubo.acquisition": [[12, 0, 0, "-", "acquisition_function"], [12, 0, 0, "-", "expected_improvement"], [12, 0, 0, "-", "upper_confidence_bound"]], "nubo.acquisition.acquisition_function": [[12, 1, 1, "", "AcquisitionFunction"]], "nubo.acquisition.expected_improvement": [[12, 1, 1, "", "ExpectedImprovement"], [12, 1, 1, "", "MCExpectedImprovement"]], "nubo.acquisition.expected_improvement.ExpectedImprovement": [[12, 2, 1, "", "eval"]], "nubo.acquisition.expected_improvement.MCExpectedImprovement": [[12, 2, 1, "", "eval"]], "nubo.acquisition.upper_confidence_bound": [[12, 1, 1, "", "MCUpperConfidenceBound"], [12, 1, 1, "", "UpperConfidenceBound"]], "nubo.acquisition.upper_confidence_bound.MCUpperConfidenceBound": [[12, 2, 1, "", "eval"]], "nubo.acquisition.upper_confidence_bound.UpperConfidenceBound": [[12, 2, 1, "", "eval"]], "nubo.models": [[13, 0, 0, "-", "fit"], [13, 0, 0, "-", "gaussian_process"]], "nubo.models.fit": [[13, 3, 1, "", "fit_gp"]], "nubo.models.gaussian_process": [[13, 1, 1, "", "GaussianProcess"]], "nubo.models.gaussian_process.GaussianProcess": [[13, 2, 1, "", "forward"]], "nubo.optimisation": [[14, 0, 0, "-", "adam"], [14, 0, 0, "-", "lbfgsb"], [14, 0, 0, "-", "multipoint"], [14, 0, 0, "-", "slsqp"], [14, 0, 0, "-", "utils"]], "nubo.optimisation.adam": [[14, 3, 1, "", "adam"]], "nubo.optimisation.lbfgsb": [[14, 3, 1, "", "lbfgsb"]], "nubo.optimisation.multipoint": [[14, 3, 1, "", "joint"], [14, 3, 1, "", "sequential"]], "nubo.optimisation.slsqp": [[14, 3, 1, "", "slsqp"]], "nubo.optimisation.utils": [[14, 3, 1, "", "gen_candidates"]], "nubo.test_functions": [[15, 0, 0, "-", "ackley"], [15, 0, 0, "-", "dixonprice"], [15, 0, 0, "-", "griewank"], [15, 0, 0, "-", "hartmann"], [15, 0, 0, "-", "levy"], [15, 0, 0, "-", "rastrigin"], [15, 0, 0, "-", "schwefel"], [15, 0, 0, "-", "sphere"], [15, 0, 0, "-", "sumsquares"], [15, 0, 0, "-", "test_functions"], [15, 0, 0, "-", "zakharov"]], "nubo.test_functions.ackley": [[15, 1, 1, "", "Ackley"]], "nubo.test_functions.dixonprice": [[15, 1, 1, "", "DixonPrice"]], "nubo.test_functions.griewank": [[15, 1, 1, "", "Griewank"]], "nubo.test_functions.hartmann": [[15, 1, 1, "", "Hartmann3D"], [15, 1, 1, "", "Hartmann6D"]], "nubo.test_functions.levy": [[15, 1, 1, "", "Levy"]], "nubo.test_functions.rastrigin": [[15, 1, 1, "", "Rastrigin"]], "nubo.test_functions.schwefel": [[15, 1, 1, "", "Schwefel"]], "nubo.test_functions.sphere": [[15, 1, 1, "", "Sphere"]], "nubo.test_functions.sumsquares": [[15, 1, 1, "", "SumSquares"]], "nubo.test_functions.test_functions": [[15, 1, 1, "", "TestFunction"]], "nubo.test_functions.zakharov": [[15, 1, 1, "", "Zakharov"]], "nubo.utils": [[16, 0, 0, "-", "latin_hypercube"], [16, 0, 0, "-", "transfrom"]], "nubo.utils.latin_hypercube": [[16, 1, 1, "", "LatinHypercubeSampling"]], "nubo.utils.latin_hypercube.LatinHypercubeSampling": [[16, 2, 1, "", "maximin"], [16, 2, 1, "", "random"]], "nubo.utils.transfrom": [[16, 3, 1, "", "normalise"], [16, 3, 1, "", "standardise"], [16, 3, 1, "", "unnormalise"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"acquisit": [0, 12, 17], "function": [0, 6, 17], "analyt": 0, "mont": 0, "carlo": 0, "asynchron": 1, "bayesian": [1, 2, 3, 5, 7, 9, 10, 18], "optimis": [1, 2, 3, 5, 6, 7, 9, 10, 14, 18], "fulli": 2, "gaussian": 2, "process": 2, "citat": 4, "select": 4, "public": 4, "us": 4, "nubo": [4, 6, 7, 8, 11, 12, 13, 14, 15, 16], "constrain": 5, "get": 6, "start": 6, "instal": 6, "toi": 6, "transpar": 7, "python": 7, "packag": [7, 11, 12, 13, 14, 15, 16], "exampl": 7, "refer": [7, 17], "parallel": [9, 10], "multi": [9, 10], "point": [9, 10, 18], "joint": 9, "sequenti": 10, "subpackag": 11, "modul": [11, 12, 13, 14, 15, 16], "content": [11, 12, 13, 14, 15, 16, 17], "submodul": [12, 13, 14, 15, 16], "acquisition_funct": 12, "expected_improv": 12, "upper_confidence_bound": 12, "model": [13, 19], "fit": 13, "gaussian_process": 13, "adam": 14, "lbfgsb": 14, "multipoint": 14, "slsqp": 14, "util": [14, 16], "test_funct": 15, "acklei": 15, "dixonpric": 15, "griewank": 15, "hartmann": 15, "levi": 15, "rastrigin": 15, "schwefel": 15, "sphere": 15, "sumsquar": 15, "zakharov": 15, "latin_hypercub": 16, "transfrom": 16, "overview": 17, "choic": 17, "singl": 18, "surrog": 19}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"Acquisition functions": [[0, "acquisition-functions"]], "Analytical acquisition functions": [[0, "analytical-acquisition-functions"]], "Monte Carlo acquisition functions": [[0, "monte-carlo-acquisition-functions"]], "Asynchronous Bayesian Optimisation": [[1, "Asynchronous-Bayesian-Optimisation"]], "Fully Bayesian Gaussian Process for Bayesian Optimisation": [[2, "Fully-Bayesian-Gaussian-Process-for-Bayesian-Optimisation"]], "Bayesian Optimisation": [[3, "bayesian-optimisation"]], "Citation": [[4, "citation"]], "Selected publications using NUBO": [[4, "selected-publications-using-nubo"]], "Constrained Bayesian Optimisation": [[5, "Constrained-Bayesian-Optimisation"]], "Get started": [[6, "Get-started"]], "Installing NUBO": [[6, "Installing-NUBO"]], "Optimising a toy function with NUBO": [[6, "Optimising-a-toy-function-with-NUBO"]], "NUBO: a transparent python package for Bayesian Optimisation": [[7, "nubo-a-transparent-python-package-for-bayesian-optimisation"]], "NUBO": [[7, "nubo"]], "NUBO:": [[7, null]], "Examples": [[7, "examples"]], "Examples:": [[7, null]], "Package reference": [[7, "package-reference"]], "Package reference:": [[7, null]], "nubo": [[8, "nubo"]], "Parallel multi-point joint Bayesian Optimisation": [[9, "Parallel-multi-point-joint-Bayesian-Optimisation"]], "Parallel multi-point sequential Bayesian Optimisation": [[10, "Parallel-multi-point-sequential-Bayesian-Optimisation"]], "nubo package": [[11, "nubo-package"]], "Subpackages": [[11, "subpackages"]], "Module contents": [[11, "module-nubo"], [12, "module-nubo.acquisition"], [13, "module-nubo.models"], [14, "module-nubo.optimisation"], [15, "module-nubo.test_functions"], [16, "module-nubo.utils"]], "nubo.acquisition package": [[12, "nubo-acquisition-package"]], "Submodules": [[12, "submodules"], [13, "submodules"], [14, "submodules"], [15, "submodules"], [16, "submodules"]], "nubo.acquisition.acquisition_function module": [[12, "module-nubo.acquisition.acquisition_function"]], "nubo.acquisition.expected_improvement module": [[12, "module-nubo.acquisition.expected_improvement"]], "nubo.acquisition.upper_confidence_bound module": [[12, "module-nubo.acquisition.upper_confidence_bound"]], "nubo.models package": [[13, "nubo-models-package"]], "nubo.models.fit module": [[13, "module-nubo.models.fit"]], "nubo.models.gaussian_process module": [[13, "module-nubo.models.gaussian_process"]], "nubo.optimisation package": [[14, "nubo-optimisation-package"]], "nubo.optimisation.adam module": [[14, "module-nubo.optimisation.adam"]], "nubo.optimisation.lbfgsb module": [[14, "module-nubo.optimisation.lbfgsb"]], "nubo.optimisation.multipoint module": [[14, "module-nubo.optimisation.multipoint"]], "nubo.optimisation.slsqp module": [[14, "module-nubo.optimisation.slsqp"]], "nubo.optimisation.utils module": [[14, "module-nubo.optimisation.utils"]], "nubo.test_functions package": [[15, "nubo-test-functions-package"]], "nubo.test_functions.ackley module": [[15, "module-nubo.test_functions.ackley"]], "nubo.test_functions.dixonprice module": [[15, "module-nubo.test_functions.dixonprice"]], "nubo.test_functions.griewank module": [[15, "module-nubo.test_functions.griewank"]], "nubo.test_functions.hartmann module": [[15, "module-nubo.test_functions.hartmann"]], "nubo.test_functions.levy module": [[15, "module-nubo.test_functions.levy"]], "nubo.test_functions.rastrigin module": [[15, "module-nubo.test_functions.rastrigin"]], "nubo.test_functions.schwefel module": [[15, "module-nubo.test_functions.schwefel"]], "nubo.test_functions.sphere module": [[15, "module-nubo.test_functions.sphere"]], "nubo.test_functions.sumsquares module": [[15, "module-nubo.test_functions.sumsquares"]], "nubo.test_functions.test_functions module": [[15, "module-nubo.test_functions.test_functions"]], "nubo.test_functions.zakharov module": [[15, "module-nubo.test_functions.zakharov"]], "nubo.utils package": [[16, "nubo-utils-package"]], "nubo.utils.latin_hypercube module": [[16, "module-nubo.utils.latin_hypercube"]], "nubo.utils.transfrom module": [[16, "module-nubo.utils.transfrom"]], "Overview": [[17, "overview"]], "Contents": [[17, "contents"]], "Choice of acquisition function": [[17, "choice-of-acquisition-function"]], "References": [[17, "references"]], "Single-point Bayesian Optimisation": [[18, "Single-point-Bayesian-Optimisation"]], "Surrogate models": [[19, "surrogate-models"]]}, "indexentries": {"module": [[11, "module-nubo"], [12, "module-nubo.acquisition"], [12, "module-nubo.acquisition.acquisition_function"], [12, "module-nubo.acquisition.expected_improvement"], [12, "module-nubo.acquisition.upper_confidence_bound"], [13, "module-nubo.models"], [13, "module-nubo.models.fit"], [13, "module-nubo.models.gaussian_process"], [14, "module-nubo.optimisation"], [14, "module-nubo.optimisation.adam"], [14, "module-nubo.optimisation.lbfgsb"], [14, "module-nubo.optimisation.multipoint"], [14, "module-nubo.optimisation.slsqp"], [14, "module-nubo.optimisation.utils"], [15, "module-nubo.test_functions"], [15, "module-nubo.test_functions.ackley"], [15, "module-nubo.test_functions.dixonprice"], [15, "module-nubo.test_functions.griewank"], [15, "module-nubo.test_functions.hartmann"], [15, "module-nubo.test_functions.levy"], [15, "module-nubo.test_functions.rastrigin"], [15, "module-nubo.test_functions.schwefel"], [15, "module-nubo.test_functions.sphere"], [15, "module-nubo.test_functions.sumsquares"], [15, "module-nubo.test_functions.test_functions"], [15, "module-nubo.test_functions.zakharov"], [16, "module-nubo.utils"], [16, "module-nubo.utils.latin_hypercube"], [16, "module-nubo.utils.transfrom"]], "nubo": [[11, "module-nubo"]], "acquisitionfunction (class in nubo.acquisition.acquisition_function)": [[12, "nubo.acquisition.acquisition_function.AcquisitionFunction"]], "expectedimprovement (class in nubo.acquisition.expected_improvement)": [[12, "nubo.acquisition.expected_improvement.ExpectedImprovement"]], "mcexpectedimprovement (class in nubo.acquisition.expected_improvement)": [[12, "nubo.acquisition.expected_improvement.MCExpectedImprovement"]], "mcupperconfidencebound (class in nubo.acquisition.upper_confidence_bound)": [[12, "nubo.acquisition.upper_confidence_bound.MCUpperConfidenceBound"]], "upperconfidencebound (class in nubo.acquisition.upper_confidence_bound)": [[12, "nubo.acquisition.upper_confidence_bound.UpperConfidenceBound"]], "eval() (nubo.acquisition.expected_improvement.expectedimprovement method)": [[12, "nubo.acquisition.expected_improvement.ExpectedImprovement.eval"]], "eval() (nubo.acquisition.expected_improvement.mcexpectedimprovement method)": [[12, "nubo.acquisition.expected_improvement.MCExpectedImprovement.eval"]], "eval() (nubo.acquisition.upper_confidence_bound.mcupperconfidencebound method)": [[12, "nubo.acquisition.upper_confidence_bound.MCUpperConfidenceBound.eval"]], "eval() (nubo.acquisition.upper_confidence_bound.upperconfidencebound method)": [[12, "nubo.acquisition.upper_confidence_bound.UpperConfidenceBound.eval"]], "nubo.acquisition": [[12, "module-nubo.acquisition"]], "nubo.acquisition.acquisition_function": [[12, "module-nubo.acquisition.acquisition_function"]], "nubo.acquisition.expected_improvement": [[12, "module-nubo.acquisition.expected_improvement"]], "nubo.acquisition.upper_confidence_bound": [[12, "module-nubo.acquisition.upper_confidence_bound"]], "gaussianprocess (class in nubo.models.gaussian_process)": [[13, "nubo.models.gaussian_process.GaussianProcess"]], "fit_gp() (in module nubo.models.fit)": [[13, "nubo.models.fit.fit_gp"]], "forward() (nubo.models.gaussian_process.gaussianprocess method)": [[13, "nubo.models.gaussian_process.GaussianProcess.forward"]], "nubo.models": [[13, "module-nubo.models"]], "nubo.models.fit": [[13, "module-nubo.models.fit"]], "nubo.models.gaussian_process": [[13, "module-nubo.models.gaussian_process"]], "adam() (in module nubo.optimisation.adam)": [[14, "nubo.optimisation.adam.adam"]], "gen_candidates() (in module nubo.optimisation.utils)": [[14, "nubo.optimisation.utils.gen_candidates"]], "joint() (in module nubo.optimisation.multipoint)": [[14, "nubo.optimisation.multipoint.joint"]], "lbfgsb() (in module nubo.optimisation.lbfgsb)": [[14, "nubo.optimisation.lbfgsb.lbfgsb"]], "nubo.optimisation": [[14, "module-nubo.optimisation"]], "nubo.optimisation.adam": [[14, "module-nubo.optimisation.adam"]], "nubo.optimisation.lbfgsb": [[14, "module-nubo.optimisation.lbfgsb"]], "nubo.optimisation.multipoint": [[14, "module-nubo.optimisation.multipoint"]], "nubo.optimisation.slsqp": [[14, "module-nubo.optimisation.slsqp"]], "nubo.optimisation.utils": [[14, "module-nubo.optimisation.utils"]], "sequential() (in module nubo.optimisation.multipoint)": [[14, "nubo.optimisation.multipoint.sequential"]], "slsqp() (in module nubo.optimisation.slsqp)": [[14, "nubo.optimisation.slsqp.slsqp"]], "ackley (class in nubo.test_functions.ackley)": [[15, "nubo.test_functions.ackley.Ackley"]], "dixonprice (class in nubo.test_functions.dixonprice)": [[15, "nubo.test_functions.dixonprice.DixonPrice"]], "griewank (class in nubo.test_functions.griewank)": [[15, "nubo.test_functions.griewank.Griewank"]], "hartmann3d (class in nubo.test_functions.hartmann)": [[15, "nubo.test_functions.hartmann.Hartmann3D"]], "hartmann6d (class in nubo.test_functions.hartmann)": [[15, "nubo.test_functions.hartmann.Hartmann6D"]], "levy (class in nubo.test_functions.levy)": [[15, "nubo.test_functions.levy.Levy"]], "rastrigin (class in nubo.test_functions.rastrigin)": [[15, "nubo.test_functions.rastrigin.Rastrigin"]], "schwefel (class in nubo.test_functions.schwefel)": [[15, "nubo.test_functions.schwefel.Schwefel"]], "sphere (class in nubo.test_functions.sphere)": [[15, "nubo.test_functions.sphere.Sphere"]], "sumsquares (class in nubo.test_functions.sumsquares)": [[15, "nubo.test_functions.sumsquares.SumSquares"]], "testfunction (class in nubo.test_functions.test_functions)": [[15, "nubo.test_functions.test_functions.TestFunction"]], "zakharov (class in nubo.test_functions.zakharov)": [[15, "nubo.test_functions.zakharov.Zakharov"]], "nubo.test_functions": [[15, "module-nubo.test_functions"]], "nubo.test_functions.ackley": [[15, "module-nubo.test_functions.ackley"]], "nubo.test_functions.dixonprice": [[15, "module-nubo.test_functions.dixonprice"]], "nubo.test_functions.griewank": [[15, "module-nubo.test_functions.griewank"]], "nubo.test_functions.hartmann": [[15, "module-nubo.test_functions.hartmann"]], "nubo.test_functions.levy": [[15, "module-nubo.test_functions.levy"]], "nubo.test_functions.rastrigin": [[15, "module-nubo.test_functions.rastrigin"]], "nubo.test_functions.schwefel": [[15, "module-nubo.test_functions.schwefel"]], "nubo.test_functions.sphere": [[15, "module-nubo.test_functions.sphere"]], "nubo.test_functions.sumsquares": [[15, "module-nubo.test_functions.sumsquares"]], "nubo.test_functions.test_functions": [[15, "module-nubo.test_functions.test_functions"]], "nubo.test_functions.zakharov": [[15, "module-nubo.test_functions.zakharov"]], "latinhypercubesampling (class in nubo.utils.latin_hypercube)": [[16, "nubo.utils.latin_hypercube.LatinHypercubeSampling"]], "maximin() (nubo.utils.latin_hypercube.latinhypercubesampling method)": [[16, "nubo.utils.latin_hypercube.LatinHypercubeSampling.maximin"]], "normalise() (in module nubo.utils.transfrom)": [[16, "nubo.utils.transfrom.normalise"]], "nubo.utils": [[16, "module-nubo.utils"]], "nubo.utils.latin_hypercube": [[16, "module-nubo.utils.latin_hypercube"]], "nubo.utils.transfrom": [[16, "module-nubo.utils.transfrom"]], "random() (nubo.utils.latin_hypercube.latinhypercubesampling method)": [[16, "nubo.utils.latin_hypercube.LatinHypercubeSampling.random"]], "standardise() (in module nubo.utils.transfrom)": [[16, "nubo.utils.transfrom.standardise"]], "unnormalise() (in module nubo.utils.transfrom)": [[16, "nubo.utils.transfrom.unnormalise"]]}})