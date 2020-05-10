import atexit
import fnmatch
import json
import os
import numpy as np
import sys
import re
import random
import shutil
import string
import time
import traceback
import warnings

import torch
from experiment_setup.experiment_base.experiment_log import PytorchExperimentLogger
#from .experiment_base import Experiment
#sys.path.append("..")
#sys.path.append("..")
from util.config import  Config
from util.util import ResultElement, ResultLogDict, name_and_iter_to_filename
from util.SourcePacker import SourcePacker
from util.pytorchutils import set_seed


class PytorchExperiment(object):
    """
    A PytorchExperiment extends the basic
    functionality of the :class:`.Experiment` class with
    convenience features for PyTorch (and general logging) such as creating a folder structure,
    saving, plotting results and checkpointing your experiment.

    The basic life cycle of a PytorchExperiment is the same as
    :class:`.Experiment`::

        setup()
        prepare()

        for epoch in n_epochs:
            train()
            validate()

        end()

    where the distinction between the first two is that between them
    PytorchExperiment will automatically restore checkpoints and save the
    :attr:`_config_raw` in :meth:`._setup_internal`. Please see below for more
    information on this.
    To get your own experiment simply inherit from the PytorchExperiment and
    overwrite the :meth:`.setup`, :meth:`.prepare`, :meth:`.train`,
    :meth:`.validate` method (or you can use the `very` experimental decorator
    :func:`.experimentify` to convert your class into a experiment).
    Then you can run your own experiment by calling the :meth:`.run` method.

    Internally PytorchExperiment will provide a number of member variables which
    you can access.

        - n_epochs
            Number of epochs.
        - exp_name
            Name of your experiment.
        - config
            The (initialized) :class:`.Config` of your experiment. You can
            access the uninitialized one via :attr:`_config_raw`.
        - result
            A dict in which you can store your result values. If a
            :class:`.PytorchExperimentLogger` is used, results will be a
            :class:`.ResultLogDict` that directly automatically writes to a file
            and also stores the N last entries for each key for quick access
            (e.g. to quickly get the running mean).
        - vlog (if use_visdomlogger is True)
            A :class:`.PytorchVisdomLogger` instance which can log your results
            to a running visdom server. Start the server via
            :code:`python -m visdom.server` or pass :data:`auto_start=True` in
            the :attr:`visdomlogger_kwargs`.
        - elog (if use_explogger is True)
            A :class:`.PytorchExperimentLogger` instance which can log your
            results to a given folder.
        - tlog (if use_telegrammessagelogger is True)
            A :class:`.TelegramMessageLogger` instance which can send the results to
            your telegram account
        - clog
            A :class:`.CombinedLogger` instance which logs to all loggers with
            different frequencies (specified with the :attr:`_c_freq` for each
            logger where 1 means every time and N means every Nth time,
            e.g. if you only want to send stuff to Visdom every 10th time).

    The most important attribute is certainly :attr:`.config`, which is the
    initialized :class:`.Config` for the experiment. To understand how it needs
    to be structured to allow for automatic instantiation of types, please refer
    to its documentation. If you decide not to use this functionality,
    :attr:`config` and :attr:`_config_raw` are identical. **Beware however that
    by default the Pytorchexperiment only saves the raw config** after
    :meth:`.setup`. If you modify :attr:`config` during setup, make sure
    to implement :meth:`._setup_internal` yourself should you want the modified
    config to be saved::

        def _setup_internal(self):

            super(YourExperiment, self)._setup_internal() # calls .prepare_resume()
            self.elog.save_config(self.config, "config")

    Args:
        config (dict or Config): Configures your experiment. If :attr:`name`,
            :attr:`n_epochs`, :attr:`seed`, :attr:`base_dir` are given in the
            config, it will automatically
            overwrite the other args/kwargs with the values from the config.
            In addition (defined by :attr:`parse_config_sys_argv`) the config
            automatically parses the argv arguments and updates its values if a
            key matches a console argument.
        name (str):
            The name of the PytorchExperiment.
        n_epochs (int): The number of epochs (number of times the training
            cycle will be executed).
        seed (int): A random seed (which will set the random, numpy and
            torch seed).
        base_dir (str): A base directory in which the experiment result folder
            will be created.
        globs: The :func:`globals` of the script which is run. This is necessary
            to get and save the executed files in the experiment folder.
        resume (str or PytorchExperiment): Another PytorchExperiment or path to
            the result dir from another PytorchExperiment from which it will
            load the PyTorch modules and other member variables and resume
            the experiment.
        ignore_resume_config (bool): If :obj:`True` it will not resume with the
            config from the resume experiment but take the current/own config.
        resume_save_types (list or tuple): A list which can define which values
            to restore when resuming. Choices are:

                - "model" <-- Pytorch models
                - "optimizer" <-- Optimizers
                - "simple" <-- Simple python variables (basic types and lists/tuples
                - "th_vars" <-- torch tensors/variables
                - "results" <-- The result dict

     """

    def __init__(self,
                 config=None,
                 name=None,
                 n_epochs=None,
                 seed=None,
                 base_dir=None,
                 globs=None,
                 parse_config_sys_argv=True,
                 checkpoint_to_cpu=True,
                 exp_ID='000',
                  ):

        # super(PytorchExperiment, self).__init__()
        # Experiment.__init__(self)
        self.exp_ID = exp_ID


        self._epoch_idx = 0

        self._config_raw = None
        if isinstance(config, str):
            self._config_raw = Config(file_=config, update_from_argv=parse_config_sys_argv)
        elif isinstance(config, Config):
            self._config_raw = Config(config=config, update_from_argv=parse_config_sys_argv)
        elif isinstance(config, dict):
            self._config_raw = Config(config=config, update_from_argv=parse_config_sys_argv)
        else:
            self._config_raw = Config(update_from_argv=parse_config_sys_argv)

        self.n_epochs = n_epochs
        if 'n_epochs' in self._config_raw:
            self.n_epochs = self._config_raw["n_epochs"]
        if self.n_epochs is None:
            self.n_epochs = 0

        self._seed = seed
        if 'seed' in self._config_raw:
            self._seed = self._config_raw.seed
        if self._seed is None:
            random_data = os.urandom(4)
            seed = int.from_bytes(random_data, byteorder="big")
            self._config_raw.seed = seed
            self._seed = seed

        self.exp_name = name
        if 'name' in self._config_raw:
            self.exp_name = self._config_raw["name"]

        if 'base_dir' in self._config_raw:
            base_dir = self._config_raw["base_dir"]

        self.base_dir = os.path.join(base_dir, exp_ID+'_'+str(config.cross_vali_index)+"_"+name+time.strftime("_%y%m%d_%H%M%S", time.localtime(time.time())))
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.checkpoint_dir = os.path.join(self.base_dir, 'model')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.code_dir = os.path.join(self.base_dir, 'code')
        if not os.path.exists(self.code_dir):
            os.makedirs(self.code_dir)

        self.elog = PytorchExperimentLogger(self.base_dir)

        self._checkpoint_to_cpu = checkpoint_to_cpu
        self.results = dict()

        set_seed(self._seed)

        # self.elog.save_config(self.config, "config_pre")
        #if globs is not None: # comment out by Yukun due to error in new CRC environment
        #    zip_name = os.path.join(self.code_dir, "sources.zip")
        #    SourcePacker.zip_sources(globs, zip_name)

        # Init objects in config
        self.config = Config.init_objects(self._config_raw)

        atexit.register(self.at_exit_func)

    def run_train(self):
        """
        This method runs the Experiment. It runs through the basic lifecycle of an Experiment::

            setup()
            for epoch in n_epochs:
                train()
                validate()

            end()

        """

        try:
            self._time_start = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
            self._time_end = ""

            self.setup()
            #self._setup_internal()
            #self.prepare()

            self._exp_state = "Started"
            #self._start_internal()
            print("Experiment started.")

            for epoch in range(self._epoch_idx, self.n_epochs):
                self.train(epoch=epoch)
                self.validate(epoch=epoch)
                #self._end_epoch_internal(epoch=epoch)
                self._epoch_idx += 1

            self._exp_state = "Trained"
            print("Training complete.")

            #self.end()
            self._end_internal()
            self._exp_state = "Ended"
            print("Experiment ended.")

            self._time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self._exp_state = "Error"
            self.process_err(e)
            self._time_end = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))

            raise e

    def run_test(self, setup=True):
        """
        This method runs the Experiment.

        The test consist of an optional setup and then calls the :meth:`.test` and :meth:`.end_test`.

        Args:
            setup: If True it will execute the :meth:`.setup` and :meth:`.prepare` function similar to the run method
                before calling :meth:`.test`.

        """

        try:

            if setup:
                self.setup()
                #self._setup_internal()
                #self.prepare()

            self._exp_state = "Testing"
            print("Start test.")

            self.test()
            #self.end_test()
            self._exp_state = "Tested"

            #self._end_test_internal()

            print("Testing complete.")

        except Exception as e:

            # run_error = e
            # run_error_traceback = traceback.format_tb(e.__traceback__)
            self._exp_state = "Error"
            self.process_err(e)

            raise e

    def process_err(self, e):
        print("error:".join(traceback.format_tb(e.__traceback__)))

    def setup(self):
        """Is called at the beginning of each Experiment run to setup the basic components needed for a run"""
        pass

    def train(self):
        """Is called at the beginning of each Experiment run to setup the basic components needed for a run"""
        pass

    def test(self):
        """Is called at the beginning of each Experiment run to setup the basic components needed for a run"""
        pass

    def validate(self):
        """Is called at the beginning of each Experiment run to setup the basic components needed for a run"""
        pass

    def get_pytorch_modules(self, from_config=True):
        """
        Returns all torch.nn.Modules stored in the experiment in a dict.

        Args:
            from_config (bool): Also get modules that are stored in the :attr:`.config` attribute.

        Returns:
            dict: Dictionary of PyTorch modules

        """

        pyth_modules = dict()
        for key, val in self.__dict__.items():
            if isinstance(val, torch.nn.Module):
                pyth_modules[key] = val
        if from_config:
            for key, val in self.config.items():
                if isinstance(val, torch.nn.Module):
                    if type(key) == str:
                        key = "config." + key
                    pyth_modules[key] = val
        return pyth_modules

    def get_pytorch_optimizers(self, from_config=True):
        """
        Returns all torch.optim.Optimizers stored in the experiment in a dict.

        Args:
            from_config (bool): Also get optimizers that are stored in the :attr:`.config`
                attribute.

        Returns:
            dict: Dictionary of PyTorch optimizers

        """

        pyth_optimizers = dict()
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                pyth_optimizers[key] = val
        if from_config:
            for key, val in self.config.items():
                if isinstance(val, torch.optim.Optimizer):
                    if type(key) == str:
                        key = "config." + key
                    pyth_optimizers[key] = val
        return pyth_optimizers

    def get_simple_variables(self, ignore=()):
        """
        Returns all standard variables in the experiment in a dict.
        Specifically, this looks for types :class:`int`, :class:`float`, :class:`bytes`,
        :class:`bool`, :class:`str`, :class:`set`, :class:`list`, :class:`tuple`.

        Args:
            ignore (list or tuple): Iterable of names which will be ignored

        Returns:
            dict: Dictionary of variables

        """

        simple_vars = dict()
        for key, val in self.__dict__.items():
            if key in ignore:
                continue
            if isinstance(val, (int, float, bytes, bool, str, set, list, tuple)):
                simple_vars[key] = val
        return simple_vars

    def get_pytorch_tensors(self, ignore=()):
        """
        Returns all torch.tensors in the experiment in a dict.

        Args:
            ignore (list or tuple): Iterable of names which will be ignored

        Returns:
            dict: Dictionary of PyTorch tensor

        """

        pytorch_vars = dict()
        for key, val in self.__dict__.items():
            if key in ignore:
                continue
            if torch.is_tensor(val):
                pytorch_vars[key] = val
        return pytorch_vars

    def get_pytorch_variables(self, ignore=()):
        """Same as :meth:`.get_pytorch_tensors`."""
        return self.get_pytorch_tensors(ignore)


    def save_checkpoint(self,
                        name="checkpoint",
                        save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None,
                        iter_format="{:05d}",
                        prefix=False):
        """
        Saves a current model checkpoint from the experiment.

        Args:
            name (str): The name of the checkpoint file
            save_types (list or tuple): What kind of member variables should be stored? Choices are:
                "model" <-- Pytorch models,
                "optimizer" <-- Optimizers,
                "simple" <-- Simple python variables (basic types and lists/tuples),
                "th_vars" <-- torch tensors,
                "results" <-- The result dict
            n_iter (int): Number of iterations. Together with the name, defined by the iter_format,
                a file name will be created.
            iter_format (str): Defines how the name and the n_iter will be combined.
            prefix (bool): If True, the formatted n_iter will be prepended, otherwise appended.

        """

        model_dict = {}
        optimizer_dict = {}
        simple_dict = {}
        th_vars_dict = {}
        results_dict = {}

        if "model" in save_types:
            model_dict = self.get_pytorch_modules()
        if "optimizer" in save_types:
            optimizer_dict = self.get_pytorch_optimizers()
        if "simple" in save_types:
            simple_dict = self.get_simple_variables()
        if "th_vars" in save_types:
            th_vars_dict = self.get_pytorch_variables()
        if "results" in save_types:
            results_dict = {"results": self.results}

        checkpoint_dict = {**model_dict, **optimizer_dict, **simple_dict, **th_vars_dict, **results_dict}

        #self.save_checkpoint(name=name, n_iter=n_iter, iter_format=iter_format, prefix=prefix,
        #                          move_to_cpu=self._checkpoint_to_cpu, **checkpoint_dict)
        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth.tar",
                                             iter_format=iter_format,
                                             prefix=prefix)

        if not name.endswith(".pth.tar"):
            name += ".pth.tar"
        for key, value in checkpoint_dict.items():
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                checkpoint_dict[key] = value.state_dict()

        checkpoint_file = os.path.join(self.checkpoint_dir, name)

        def to_cpu(obj):
            if hasattr(obj, "cpu"):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {key: to_cpu(val) for key, val in obj.items()}
            else:
                return obj

        move_to_cpu = True  ## xxw need debug
        if move_to_cpu:
            torch.save(to_cpu(checkpoint_dict), checkpoint_file)
        else:
            torch.save(checkpoint_dict, checkpoint_file)

    def update_attributes(self, var_dict, ignore=()):
        """
        Updates the member attributes with the attributes given in the var_dict

        Args:
            var_dict (dict): dict in which the update values stored. If a key matches a member attribute name
                the member attribute will be updated
            ignore (list or tuple): iterable of keys to ignore

        """
        for key, val in var_dict.items():
            if key == "results":
                self.results.load(val)
                continue
            if key in ignore:
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    def _end_internal(self):
        """Ends the experiment and stores the final results/checkpoint"""
        if isinstance(self.results, ResultLogDict):
            self.results.close()
        self.save_end_checkpoint()
        self.elog.print("Experiment ended. Checkpoints stored =)")

    def update_model(self, original_model, update_dict, exclude_layers=(), do_warnings=True):
        # also allow loading of partially pretrained net
        model_dict = original_model.state_dict()

        # 1. Give warnings for unused update values
        unused = set(update_dict.keys()) - set(exclude_layers) - set(model_dict.keys())
        not_updated = set(model_dict.keys()) - set(exclude_layers) - set(update_dict.keys())
        if do_warnings:
            for item in unused:
                warnings.warn("Update layer {} not used.".format(item))
            for item in not_updated:
                warnings.warn("{} layer not updated.".format(item))

        # 2. filter out unnecessary keys
        update_dict = {k: v for k, v in update_dict.items() if
                       k in model_dict and k not in exclude_layers}

        # 3. overwrite entries in the existing state dict
        model_dict.update(update_dict)

        # 4. load the new state dict
        original_model.load_state_dict(model_dict)

    def load_checkpoint_static(self, checkpoint_file, exclude_layer_dict=None, warnings=True, **kwargs):
        """
        Loads a checkpoint/dict in a given directory (using pytorch)

        Args:
            checkpoint_file: The checkpoint from which the checkpoint/dict should be loaded
            exclude_layer_dict: A dict with key 'model_name' and a list of all layers of 'model_name' which should
            not be restored
            warnings: Flag which indicates if method should warn if not everything went perfectlys
            **kwargs: dict which is actually loaded (key=name (used to save the checkpoint) , value=variable to be
            loaded/ overwritten)

        Returns: The kwargs dict with the loaded/ overwritten values

        """

        if exclude_layer_dict is None:
            exclude_layer_dict = {}

        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        for key, value in kwargs.items():
            if key in checkpoint:
                if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                    exclude_layers = exclude_layer_dict.get(key, [])
                    self.update_model(value, checkpoint[key], exclude_layers, warnings)
                else:
                    kwargs[key] = checkpoint[key]

        return kwargs

    def load_checkpoint(self,
                        name="checkpoint",
                        save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None,
                        iter_format="{:05d}",
                        prefix=False,
                        path=None):
        """
        Loads a checkpoint and restores the experiment.

        Make sure you have your torch stuff already on the right devices beforehand,
        otherwise this could lead to errors e.g. when making a optimizer step
        (and for some reason the Adam states are not already on the GPU:
        https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3 )

        Args:
            name (str): The name of the checkpoint file
            save_types (list or tuple): What kind of member variables should be loaded? Choices are:
                "model" <-- Pytorch models,
                "optimizer" <-- Optimizers,
                "simple" <-- Simple python variables (basic types and lists/tuples),
                "th_vars" <-- torch tensors,
                "results" <-- The result dict
            n_iter (int): Number of iterations. Together with the name, defined by the iter_format,
                a file name will be created and searched for.
            iter_format (str): Defines how the name and the n_iter will be combined.
            prefix (bool): If True, the formatted n_iter will be prepended, otherwise appended.
            path (str): If no path is given then it will take the current experiment dir and formatted
                name, otherwise it will simply use the path and the formatted name to define the
                checkpoint file.

        """
        model_dict = {}
        optimizer_dict = {}
        simple_dict = {}
        th_vars_dict = {}
        results_dict = {}

        if "model" in save_types:
            model_dict = self.get_pytorch_modules()
        if "optimizer" in save_types:
            optimizer_dict = self.get_pytorch_optimizers()
        if "simple" in save_types:
            simple_dict = self.get_simple_variables()
        if "th_vars" in save_types:
            th_vars_dict = self.get_pytorch_variables()
        if "results" in save_types:
            results_dict = {"results": self.results}

        checkpoint_dict = {**model_dict, **optimizer_dict, **simple_dict, **th_vars_dict, **results_dict}

        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth.tar",
                                             iter_format=iter_format,
                                             prefix=prefix)

        checkpoint_path = os.path.join(path, name)
        if checkpoint_path.endswith("/"):
            checkpoint_path = checkpoint_path[:-1]
        #restore_dict = self.load_checkpoint_static(checkpoint_file=checkpoint_path, **checkpoint_dict)
        restore_dict = self.load_checkpoint_static(checkpoint_path, **checkpoint_dict)

        self.update_attributes(restore_dict)


    def at_exit_func(self):
        """
        Stores the results and checkpoint at the end (if not already stored).
        This method is also called if an error occurs.
        """

        if self._exp_state not in ("Ended", "Tested"):
            if isinstance(self.results, ResultLogDict):
                self.results.print_to_file("]")
            self.save_checkpoint(name="checkpoint_exit-" + self._exp_state)
            self.elog.print("Experiment exited. Checkpoints stored =)")
        time.sleep(2)  # allow checkpoint saving to finish. We need a better solution for this :D


    def save_temp_checkpoint(self):
        """Saves the current checkpoint as checkpoint_current."""
        self.save_checkpoint(name="checkpoint_current")

    def save_end_checkpoint(self):
        """Saves the current checkpoint as checkpoint_last."""
        self.save_checkpoint(name="checkpoint_last")


