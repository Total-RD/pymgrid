import json
import pandas as pd

from collections import UserDict, UserList
from pymgrid.modules.base import BaseMicrogridModule


class Container(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir_additions = self.dir_additions()

    @property
    def containers(self):
        """
        View of this container's containers.

        Returns
        -------
        containers : dict-like
            View of containers.
        """
        return self

    def to_list(self):
        """
        Get the modules as a list.

        Returns
        -------
        l : list of modules
            List of modules

        """
        l = []
        for _, raw_container in self.containers.items():
            l.extend(raw_container.to_list())
        return l

    def to_dict(self):
        """
        Get the modules as a dictionary.

        Returns
        -------
        d : dict[str, list[module]]
            Dictionary with module names as keys, modules as values.

        """
        d = dict()
        for k, raw_container in self.containers.items():
            d.update(raw_container)
        return d

    def to_tuples(self):
        """
        Get the modules in (name, module) pairs.

        Returns
        -------
        tups : list of tuples: (name, module)
            Module names and modules.

        """
        l = []
        for name, modules in self.iterdict():
            tups = list(zip([name] * len(modules), modules))
            l.extend(tups)
        return l

    def iterlist(self):
        """
        Iterable of the container's modules as a list.

        Returns
        -------
        iter : generator
            Iterator of modules.

        """
        for module in self.to_list():
            yield module

    def iterdict(self):
        """
        Iterable of the container's modules as a dict.

        Returns
        -------
        iter : generator
            Iterator of (name, module) pairs.

        """
        for name, modules in self.to_dict().items():
            yield name, modules

    def get_attrs(self, *attrs, unique=False, as_pandas=True):
        """
        Get module attributes as a dictionary or pandas object.

        If ``unique``, checks that the value is unique for all modules and returns only the unique value.
        Otherwise, returns the values for all modules.

        If ``as_pandas``, returns either a pd.Series (if ``unique``) or pd.DataFrame of attributes.

        Parameters
        ----------
        attrs : str
            Names of module attributes to return.

        unique : bool, default False
            Whether to check for and return a single, unique value for an attribute.
            If different modules have different values, a ``ValueError`` will be raised.

        as_pandas : bool, default True
            Whether to return either a pd.Series or pd.DataFrame.
            If True, the return value will be a pd.Series if ``unique`` and a pd.DataFrame otherwise.
            If False, returns a dict.

        .. note::
            If only some modules have a particular attribute, ``get_attrs`` will not raise an error.

            If ``unique``, the unique value of the modules containing the value will be returned.
            Otherwise, ``NotImplemented`` will fill in missing values.

        Returns
        -------
        d : dict or pd.DataFrame or pd.Series
            * Returns dict if ``as_pandas`` is False.

            * Otherwise, returns a pd.Series if ``unique`` and a pd.DataFrame otherwise.

        Raises
        ------
        ValueError
            * If ``attrs`` is empty, or
            * ``unique`` is True and non-unique values are found.

        AttributeError
            If no module has the particular attribute.

            .. warning::
            This check is not performed if both ``unique`` and ``as_pandas`` are False.

        """
        if not attrs:
            raise ValueError('Missing attrs to get.')

        d = dict()
        for k, raw_container in self.containers.items():
            d.update({
                name: [{attr: getattr(module, attr, NotImplemented) for attr in attrs} for module in module_list]
                for name, module_list in raw_container.items()
            })

        if not (unique or as_pandas):
            return d

        d_df = pd.DataFrame({(name, num): subdict for name, module_list in d.items()
                            for num, subdict in enumerate(module_list)}).T

        bad_keys = []
        uniques, nonuniques = {}, []

        for k, v in d_df.items():
            not_notimplemented = v[v != NotImplemented]
            unique_items = not_notimplemented.drop_duplicates().values

            try:
                unique_item = unique_items.item()
            except ValueError:
                if len(unique_items) == 0:
                    # Only values were NotImplemented
                    bad_keys.append(k)
                else:
                    nonuniques.append(k)
            else:
                uniques[k] = unique_item

        if len(bad_keys):
            raise AttributeError(f'No values found for key(s) {bad_keys}')

        if unique:
            if len(nonuniques):
                raise ValueError(f"Attribute(s) {nonuniques} have non-unique values, cannot return single unique value.")

            if not as_pandas:
                return uniques

            return pd.Series(uniques, dtype=None if len(uniques) else float)

        if as_pandas:
            return d_df

        return d

    def set_attrs(self, attr_dict, **attrs):
        """
        Set the value of an attribute in all modules containing that attribute.

        Does nothing in modules not already containing the attribute.

        You may either pass attr_dict, a dict of attributes, or pass attributes as keyword arguments.

        Parameters
        ----------
        attr_dict : dict
            Key-value pairs or attributes to set. Keys are attribute names values are the values
            to set attribute to.
        attrs : dict
            Key-value pairs or attributes to set. Keys are attribute names values are the values
            to set attribute to.

        Raises
        ------
        AttributeError
            If no module has the attribute.

        """
        if attr_dict:
            if attrs:
                raise ValueError('Both attr_dict and keyword arguments were passed.')
        elif attrs:
            attr_dict = attrs
        else:
            raise ValueError('Missing attributes to set.')

        for attr_name, value in attr_dict.items():
            self._set_attr(attr_name, value)

    def _set_attr(self, attr_name, value):
        set_at_least_one = False

        for module in self._modules.iterlist():
            if not hasattr(module, attr_name):
                continue

            setattr(module, attr_name, value)
            set_at_least_one = True

        if not set_at_least_one:
            raise AttributeError(f"No module has attribute '{attr_name}'.")

    def dir_additions(self):
        """
        :meta private:
        """
        additions = set(self.keys())
        for x in self.values():
            try:
                additions.update(x.dir_additions)
            except AttributeError:
                pass
        return additions

    def __getitem__(self, item):
        if item == 'data' or item == 'module_dict':
            raise KeyError(item)
        try:
            return self.data[item]
        except KeyError:
            try:
                return self.to_dict()[item]
            except KeyError:
                raise KeyError(item)

    def __getattr__(self, item):
        if item == 'data' or item.startswith('__') or item not in dir(self):
            raise AttributeError(item)
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __len__(self):
        return sum(len(v) for k, v in self.containers.items())

    def __repr__(self):
        try:
            return json.dumps(self.to_dict(), indent=2, default=str)
        except TypeError:
            return super().__repr__()

    def __dir__(self):
        rv = set(super().__dir__())
        rv = rv | self.dir_additions
        return sorted(rv)

    def __contains__(self, item):
        return item in self.data.keys() or item in self.dir_additions


class ModuleContainer(Container):
    """
    Container of modules.

    Allows for indexing and viewing of a microgrids module's in various ways.
    """
    """
    Container of modules. Allows for indexing/getting of the modules in various ways.
    Modules are stored at the lowest level: self._containers[('fixed', 'source')] = [Genset, Grid], for example.
    These modules, however, can be accessed in many different ways:

        container.fixed
            All fixed modules
        container.flex
            All flex modules
        container.sources
             ll source modules
        container.sinks
             All sink modules
        container.fixed.sources
            All fixed source modules
        container.sources.fixed
            Same as above
        container.fixed.sinks
            All fixed sink modules
        container.sinks.fixed
            Same as above

        Modules can also be accessed directly:
        container.genset
            All modules named genset

        container.big_battery
            Modules passed with custom name big_battery

        Each level can be iterated on, both by calling .items() or by iterating through the modules directly with .iterlist()
        For example, container.sinks.iterlist() returns an iterator of all the sinks, without their names.

    """
    def __init__(self, modules):
        """

        :param modules: list-like. List of _modules or tuples. Latter case: tup(str, Module); str to define name of module
            and second element is the module.

        """
        self._containers = get_subcontainers(modules)
        midlevels = self._set_midlevel()
        self._types_by_name = self._get_types_by_name()
        super().__init__(**midlevels)

    def _get_types_by_name(self):
        return {name: container_type for container_type, container in self._containers.items() for name in container}

    def _set_midlevel(self):
        midlevels = dict()
        for key, subcontainer in self._containers.items():
            fixed_or_flex, source_sink_both = key

            if fixed_or_flex in midlevels:
                midlevels[fixed_or_flex][source_sink_both] = subcontainer
            else:
                midlevels[fixed_or_flex] = {source_sink_both: subcontainer}

            if source_sink_both in midlevels:
                midlevels[source_sink_both][fixed_or_flex] = subcontainer
            else:
                midlevels[source_sink_both] = {fixed_or_flex: subcontainer}
        midlevels = {k: Container(**v) for k, v in midlevels.items()}
        return midlevels

    def names(self):
        return list(self._types_by_name.keys())

    @property
    def containers(self):
        return self._containers


class ModuleList(UserList):
    def item(self):
        """
        Get the value of a singleton list.

        Returns
        -------
        module : BaseMicrogridModule
            Item in a singleton list.

        Raises
        ------
        ValueError :
            If there is more than one item in the list.

        """
        if len(self) != 1:
            raise ValueError("Can only convert a ModuleList of length one to a scalar")
        return self[0]

    def to_list(self):
        """
        :meta private:

        Function to be compatible with Container API.

        """
        return self


def get_subcontainers(modules):
    """
    :meta private:
    """
    source_sink_keys = ('sources', 'sinks', 'source_and_sinks')
    fixed = {k: dict() for k in source_sink_keys}
    flex = {k: dict() for k in source_sink_keys}
    controllable = {k: dict() for k in source_sink_keys}

    module_names = dict()

    for module in modules:
        try:  # module is a tuple
            module_name, module = module
            fixed_flex_controllable = module.__class__.module_type[1]
        except TypeError:  # module is a module
            try:
                module_name, fixed_flex_controllable = module.__class__.module_type
            except TypeError:
                raise NotImplementedError(
                    f'Must define the class attribute module_type for class {module.__class__.__name__}')

        assert isinstance(module, BaseMicrogridModule), 'Module must inherit from BaseMicrogridModule.'
        assert module.is_sink or module.is_source, 'Module must be sink or source (or both).'

        source_sink_both = 'source_and_sinks' if module.is_sink and module.is_source else \
            'sources' if module.is_source else 'sinks'

        if fixed_flex_controllable == 'fixed':
            d = fixed
        elif fixed_flex_controllable == 'flex':
            d = flex
        elif fixed_flex_controllable == 'controllable':
            d = controllable
        else:
            raise TypeError(f'Cannot parse fixed_flex_controllable from module type {module.__class__.module_type}')

        try:
            module_names[module_name] = (fixed_flex_controllable, source_sink_both)
        except KeyError:
            raise NameError(
                f'Attempted to add module {module_name} of type {(fixed_flex_controllable, source_sink_both)}, '
                f'but there is an identically named module of type {module_names[module_name]}.')

        try:
            d[source_sink_both][module_name].append(module)
        except KeyError:
            d[source_sink_both][module_name] = ModuleList([module])
        module.name = (module_name, len(d[source_sink_both][module_name]) - 1)

    modules_dict = dict(fixed=fixed,
                        flex=flex,
                        controllable=controllable)

    containers = {(ffs, source_sink_both): Container(modules_dict[ffs][source_sink_both])
                  for ffs in ('fixed', 'flex', 'controllable')
                  for source_sink_both in source_sink_keys}

    return containers
