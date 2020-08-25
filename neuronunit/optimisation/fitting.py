try:
    from exceptions import Exception
except:
    pass # python 3

import multiprocessing
import random
from time import time

import numpy as np
import quantities as pq
from deap import base, creator
from deap import tools
from olfactorybulb.neuronunit.tests.publications import *
from olfactorybulb.neuronunit.tests.utilities import cache
from pandas import DataFrame

from olfactorybulb.database import *
from olfactorybulb.neuronunit.models.neuron_cell import NeuronCellModel

SHOW_ERRORS = True
FAST_EVAL = False

class CellFitter(object):
    def __init__(self, cell_type, fitting_model_class=None):

        print("Starting "+ cell_type +" FITTER for model:", fitting_model_class)

        self.fitting_model_class = fitting_model_class
        self.cell_type = cell_type

        self.load_measurements()

        if fitting_model_class is not None:
            self.load_model_params()

        # GA classes
        creator.create("FitnessMin", base.Fitness, weights=(-1,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def load_model_params(self):
        model_cls = self.import_model_class(self.fitting_model_class)

        if hasattr(model_cls, 'params'):
            self.params = model_cls.params
        else:
            print(model_cls, 'does not have "params" attribute. The model will be evaluated, '
                             'but no fitting will be performed.')
            self.params = None


    def import_model_class(self, model_class):
        # This line takes input like: 'prev_ob_models.BhallaBower1993.isolated_cells.MC' and converts it to:
        #                         from prev_ob_models.BhallaBower1993.isolated_cells import MC
        exec ('from ' + '.'.join(model_class.split('.')[0:-1]) + " import " + model_class.split('.')[-1])

        # Import the root module
        exec ("import " + model_class.split('.')[0])

        # Return the name of the model class
        return eval(model_class.split('.')[-1])

    def pretty_pop(self):
        import pandas
        df = pandas.DataFrame([list(i) for i in self.pop], [i.fitness.values[0] for i in self.pop])
        df.columns = [p['attr'] for p in self.params]
        df = df.sort_index()
        return df

    def load_measurements(self):
        # Load ephyz measurments for the cell from DB
        self.measurements = Measurement \
            .select(Measurement) \
            .join(Property) \
            .switch(Measurement) \
            .join(Source) \
            .where((Measurement.property.type == "Electrophysiology") &
                   (Measurement.property.id.startswith(self.cell_type + '_'))) \
            .order_by(Measurement.property.id)

        # Aggregate measurments into NeuronUnit tests of properties
        self.properties = {}

        for m in self.measurements:
            # Create specific measurement test classes
            test_generic = str(m.property.test_class_generic)
            pub = str(m.source.publication_class).strip()
            class_name = test_generic + pub

            if FAST_EVAL and test_generic != "AfterHyperpolarizationTimeTest":
                continue

            if test_generic not in self.properties:
                self.properties[test_generic] = []

            # Make the created classes available globaly
            globals()[class_name] = type(class_name, (eval(pub), eval(test_generic)), {})

            # Create instances of those classes with the specific
            # measurments (against which the model will be compared)
            test_instance = eval(class_name)(observation={
                "mean": m.mean * eval(m.property.units),
                "std": m.std * eval(m.property.units),
                "n": m.n
            })

            self.properties[test_generic].append(test_instance)

        return self.properties

    def load_previous_models_as_workitems(self):
        # Load model class names from the DB
        self.model_classes = list(CellModel \
                             .select(CellModel) \
                             .where(CellModel.cell_type == self.cell_type.upper())
                             )

        # # Make the model classes loadable with 'module.module.ModelClass()'
        # for i, m in enumerate(self.model_classes):
        #     nmsp = string.join(m.isolated_model_class.split('.')[:-1], '.')
        #     cls = m.isolated_model_class.split('.')[-1]
        #
        #     import_cmd = 'from ' + nmsp + ' import ' + cls + ' as Model' + str(i)
        #     print(import_cmd)
        #     exec (import_cmd)

        # Create work item list
        self.work_items = []

        for model in self.model_classes:
            self.work_items.append({"model_class": model.isolated_model_class})

        return self.work_items

    def evaluate(self, param_set, raw_scores=False):

        # import pydevd
        # pydevd.settrace('192.168.0.100', port=4200)

        try:
            score = self.get_workitem_score({
                "model_class": self.fitting_model_class,
                "param_values": param_set
            })

            return score["model_score"],

        except:
            if SHOW_ERRORS:
                import traceback
                print(traceback.format_exc())

            raise

    def get_workitem_score(self, item):
        results = item
        results["properties"] = {}
        results["model_score"] = 0

        model_class = str(item["model_class"])

        model_cls = self.import_model_class(model_class)

        # Instantiate the model class
        cell = model_cls()

        if "param_values" in item and item["param_values"] is not None and len(item["param_values"]) > 0:
            param_values = item["param_values"]
            cell.set_model_params(param_values)
        else:
            param_values = []

        model = NeuronCellModel(cell.soma(0.5),
                                name=cell.__class__.__module__ + '.' +
                                     cell.__class__.__name__ + '|' +
                                     str(param_values))

        if FAST_EVAL:
            print("FAST_EVAL: ON. Evaluating only: ", self.properties.keys())

        # Compute the Root Mean Square error score of the model
        for prop in self.properties.keys():

            if prop not in results["properties"]:
                results["properties"][prop] = {"tests": {}, "total_n": 0, "z_score_combined": None}

            prop_tests = self.properties[prop]

            for prop_test in prop_tests:

                prop_test_result = {}
                results["properties"][prop]["tests"][prop_test.__class__.__name__] = prop_test_result

                try:
                    prediction = prop_test.generate_prediction(model)

                    if type(prediction) == str:
                        raise Exception(prediction)

                except:
                    import traceback
                    prediction = traceback.format_exc()

                    if SHOW_ERRORS:
                        print(prediction)

                prop_test_result["observation"] = prop_test.observation
                prop_test_result["prediction"] = prediction

                if type(prediction) != str:
                    z_score = (prediction - prop_test.observation["mean"]) / prop_test.observation["std"]
                    z_score = z_score.simplified
                else:
                    z_score = 6.0*pq.dimensionless  # errors are treated as 6 std deviation

                # Weigh each publication z-score by the pub sample size
                z_weighed = z_score * prop_test.observation["n"]

                prop_test_result["z_score"] = z_score
                prop_test_result["z_score_weighed"] = z_weighed

                results["properties"][prop]["total_n"] += prop_test.observation["n"]

            results["properties"][prop]["z_score_combined"] = sum([i["z_score_weighed"] for i in results["properties"][prop]["tests"].values()])
            results["properties"][prop]["z_score_combined"] /= results["properties"][prop]["total_n"]

            results["model_score"] += results["properties"][prop]["z_score_combined"].magnitude ** 2

        import math
        results["model_score"] = math.sqrt(results["model_score"])

        return results

    def get_best_score(self):

        score = self.get_workitem_score({
            "model_class": self.fitting_model_class,
            "param_values": self.best
        })

        df = []
        model = score
        row = {"model": model["model_class"]}
        for t, test in enumerate(model["properties"].keys()):
            row[test] = model["properties"][test]["z_score_combined"]
        df.append(row)

        df = DataFrame(df, [model["model_class"]])
        return df, score

    def get_workitem_scores(self):

        processes = max(1, multiprocessing.cpu_count() - 1)

        from multiprocess import Pool
        pool = Pool(processes=processes, maxtasksperchild=1)
        scores = pool.map(self.get_workitem_score, self.work_items)
        pool.close()
        pool.terminate()

        df = []
        for score in scores:
            model = score
            row = {"model": score["model_class"]}
            for t, test in enumerate(score["properties"].keys()):
                row[test] = score["properties"][test]["z_score_combined"]
            df.append(row)

        df = DataFrame(df, [score["model_class"] for score in scores])
        return df

    def get_previous_model_scores(self):

        self.load_previous_models_as_workitems()
        df = self.get_workitem_scores()

        return df

    def random_parameters(self):
        # Initial param values are uniformly distributed between the low-high bounds
        result = [random.random() for i in range(len(self.params))]

        for i, pv in enumerate(result):
            result[i] = (self.params[i]["high"] - self.params[i]["low"]) * pv + self.params[i]["low"]

        return creator.Individual(result)

    def parameter_report(self, ind):
        for pi, pv in enumerate(ind):
            param = self.params[pi]
            range = param["high"] - param["low"]
            range_loc = (pv - param["low"]) / range * 100.0
            print(("%.2f"%range_loc) + "% of range. Val: " + str(pv) + " Low: " + str(param["low"]) + " High: " + str(param["high"]) + " ATTR: " + param["attr"] + " in " + str(param["lists"]))

    def get_fitnesses(self, pop, label):
        max_wait = round(2.5 * 60) # seconds
        processes = max(1, multiprocessing.cpu_count() - 1)

        from multiprocess import Pool, TimeoutError

        pool = Pool(processes=processes, maxtasksperchild=1)
        processes = [pool.apply_async(self.evaluate, (list(ind),)) for ind in pop]

        wait_until = time() + max_wait

        fitnesses = []

        print("STARTED GEN", label, "size", len(pop))

        for pi, process in enumerate(processes):
            timeout = max(0, wait_until - time())

            try:
                result = process.get(timeout)

            except TimeoutError:
                result = 9 * 10.0,

                if SHOW_ERRORS:
                    print('Simulation timed out')

            except:
                result = 9 * 10.999,

                if SHOW_ERRORS:
                    print('Error in simulation')

            fitnesses.append(result)

            print("DONE with", pi+1, "of", len(processes), "of GEN", label, "with Fitness","%.2f" % result[0])

        pool.terminate()

        return fitnesses


    def fit(self, generation_size, number_of_generations):

        if generation_size < 5:
            raise Exception("Generation size must be 5 or more")

        if number_of_generations < 0:
            raise Exception("Number of generations must be 0 or more")

        self.pop = self.GA(self.top if hasattr(self, "top") else None, generation_size, number_of_generations)

    def GA(self, suggested_pop=None, n=30, NGEN=30):
        lows = [p["low"] for p in self.params]
        highs = [p["high"] for p in self.params]

        toolbox = base.Toolbox()
        toolbox.register("individual", self.random_parameters)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.1, low=lows, up=highs)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.1, low=lows, up=highs, indpb=0.1)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("select", tools.selNSGA2, k=int(n * 0.2))

        if suggested_pop is None:
            self.pop = toolbox.population(n=n)
        else:
            self.pop = [creator.Individual(i) for i in suggested_pop]

        MUTPB = 0.9
        F_DIVERSITY = 0.5

        n_elite_offspring = int(round(n * (1-F_DIVERSITY)))
        n_diversity_offspring = int(round(n * F_DIVERSITY / 2.0))
        n_random = n_diversity_offspring

        # Evaluate the entire population - each in a separate process
        fitnesses = self.get_fitnesses(self.pop, label="ZERO")

        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the elite parents - they already have fitness from prev-gen
            parents = toolbox.select(self.pop)

            # Elite children come from elite parents
            elite_offspring = tools.selRandom(parents, n_elite_offspring)

            # Diversity children have random parents
            diversity_offspring = tools.selRandom(self.pop, n_diversity_offspring)

            # Random offspring are randomly generated individuals
            random_offspring = toolbox.population(n=n_random)

            offspring = random_offspring + diversity_offspring + elite_offspring

            print('GEN',g+1,
                  "Parents",len(parents),
                  "Elite", len(elite_offspring),
                  "DIV",len(diversity_offspring),
                  "RND", len(random_offspring),
                  "TOT NEW", len(offspring))

            # Clone the selected individuals
            offspring = map(toolbox.clone, offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            for child in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(child)
                    del child.fitness.values

            # Evaluate the individuals without fitness value - again in separate processes
            offspring_nofitness = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = self.get_fitnesses(offspring_nofitness, label=str(g+1))

            for ind, fit in zip(offspring_nofitness, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the parents + offspring
            self.pop[:] = parents + offspring

            self.top = toolbox.select(self.pop)
            self.best = self.top[0]

            df, score = self.get_best_score()
            print(df.transpose())
            self.parameter_report(self.best)
            print('Best individual', self.best)
            print('Best fitness:', self.best.fitness.values[0])
            print("Generation", g+1, "out of", NGEN, "COMPLETE")


        return self.pop

    def clear_cache(self):
        cache.clear()

    def print_params(self, param_values, hoc_cell):
        result = ""
        indent = "  "

        lists = []
        for pv in self.params:
            lists = lists + pv["lists"]
        lists = np.unique(lists)

        for lst in lists:
            result += indent + "forsec " + lst + " {" + "\n"
            for pi, pv in enumerate(self.params):
                attr = pv["attr"]
                value = str(param_values[pi])

                if lst in pv["lists"]:
                    if attr == "diam":
                        result += indent + indent + "for i=0, n3d()-1 pt3dchange(i, diam3d(i)*" + value + ")" + "\n"
                    else:
                        result += indent + indent + attr + " = " + value + "\n"

            result += indent + "}" + "\n"

        print(result)

        return result



