import json
from collections import UserDict, UserList
from pymgrid.microgrid.modules.base import BaseMicrogridModule


class ModuleContainer(UserDict):
    """
    Container of modules. Allows for indexing/getting of the modules in various ways.
    Modules are stored at the lowest level: self._raw_containers[('fixed', 'source')] = [Genset, Grid], for example.
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
        self._raw_containers = get_subcontainers(modules)
        midlevels = self._set_midlevel()
        self._types_by_name = self._get_types_by_name()
        super().__init__(**midlevels)

    def _get_types_by_name(self):
        return {name: container_type for container_type, container in self._raw_containers.items() for name in container}

    def _set_midlevel(self):
        midlevels = dict()
        for key, subcontainer in self._raw_containers.items():
            fixed_or_flex, source_sink_both = key

            if fixed_or_flex in midlevels:
                midlevels[fixed_or_flex][source_sink_both] = subcontainer
            else:
                midlevels[fixed_or_flex] = {source_sink_both: subcontainer}

            if source_sink_both in midlevels:
                midlevels[source_sink_both][fixed_or_flex] = subcontainer
            else:
                midlevels[source_sink_both] = {fixed_or_flex: subcontainer}
        midlevels = {k: _ModulePointer(**v) for k, v in midlevels.items()}
        return midlevels

    def module_list(self):
        l = []
        for _, raw_container in self._raw_containers.items():
            l.extend(raw_container.module_list())
        return l

    def module_dict(self):
        d = dict()
        for k, raw_container in self._raw_containers.items():
            d.update(raw_container)
        return d

    def module_tuples(self):
        l = []
        for name, modules in self.iterdict():
            tups = list(zip([name]*len(modules), modules))
            l.extend(tups)
        return l

    def iterlist(self):
        for module in self.module_list():
            yield module

    def iterdict(self):
        for name, modules in self.module_dict().items():
            yield name, modules

    def names(self):
        return list(self._types_by_name.keys())

    def __getitem__(self, item):
        if item == 'data':
            raise KeyError
        try:
            return super().__getitem__(item)
        except KeyError:
            try:
                return self._raw_containers[self._types_by_name[item]][item]
            except KeyError:
                return self._raw_containers[item]

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __len__(self):
        return sum(len(v) for k, v in self._raw_containers.items())


class _ModulePointer(UserDict):
    """
    Points to fixed, flex, source, sink, etc.
    Do not initialize this directly
    """
    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def module_list(self):
        l = []
        for _, raw_container in self.data.items():
            l.extend(raw_container.module_list())
        return l

    def module_dict(self):
        d = dict()
        for k, raw_container in self.items():
            d.update(raw_container)
        return d

    def iterdict(self):
        for name, modules in self.module_dict().items():
            yield name, modules

    def iterlist(self):
        for module in self.module_list():
            yield module

    def __len__(self):
        return sum(len(v) for k, v in self.items())

    def __repr__(self):
        return json.dumps(self.module_dict(), indent=2, default=str)


def get_subcontainers(modules):
    """

    :return: List[Tuple]
        3-element tuples of (fixed_or_flex, source_or_sink, container)

    """
    source_sink_keys = ('sources' , 'sinks', 'source_and_sinks')
    fixed = {k: dict() for k in source_sink_keys}
    flex = {k: dict() for k in source_sink_keys}
    module_names = dict()

    for module in modules:
        try:  # module is a tuple
            module_name, module = module
            fixed_or_flex = module.__class__.module_type[1]
        except TypeError:  # module is a module
            try:
                module_name, fixed_or_flex = module.__class__.module_type
            except TypeError:
                raise NotImplementedError(
                    f'Must define the class attribute module_type for class {module.__class__.__name__}')

        assert isinstance(module, BaseMicrogridModule), 'Module must inherit from BaseMicrogridModule.'
        assert module.is_sink or module.is_source, 'Module must be sink or source (or both).'


        source_sink_both = 'source_and_sinks' if module.is_sink and module.is_source else \
            'sources' if module.is_source else 'sinks'

        if fixed_or_flex == 'fixed':
            d = fixed
        elif fixed_or_flex == 'flex':
            d = flex
        else:
            raise TypeError(f'Cannot parse fixed_or_flexed from module type {module.__class__.module_type}')

        try:
            module_names[module_name] = (fixed_or_flex, source_sink_both)
        except KeyError:
            raise NameError(f'Attempted to add module {module_name} of type {(fixed_or_flex, source_sink_both)}, '
                            f'but there is an identically named module of type {module_names[module_name]}.')

        try:
            d[source_sink_both][module_name].append(module)
        except KeyError:
            d[source_sink_both][module_name] = ModuleList([module])
        module.name = (module_name, len(d[source_sink_both][module_name]) - 1)

    modules_dict = dict(fixed=fixed,
                        flex=flex)

    containers = {(fixed_or_flex, source_sink_both): _ModuleSubContainer(modules_dict[fixed_or_flex][source_sink_both])
                  for fixed_or_flex in ('fixed', 'flex')
                  for source_sink_both in source_sink_keys}

    return containers


class _ModuleSubContainer(UserDict):
    """
    One of these for fixed sources, flex sources, etc.
    Do not initialize this directly
    """
    def __init__(self, modules_dict):
        fixed_or_flex, source_or_sink = self._check_modules(modules_dict)
        # self._set_module_attrs(modules_dict)
        super().__init__(**modules_dict)
        self._fixed_or_flex = fixed_or_flex
        self._source_or_sink = source_or_sink

    def _check_modules(self, modules_dict):
        fixed_or_flex = None
        source_or_sink = None

        def _get_source_sink(module):
            if module.is_source:
                if module.is_sink:
                    return 'source_and_sink'
                return 'source'
            assert module.is_sink
            return 'sink'

        for name, module_list in modules_dict.items():
            for module in module_list:
                if fixed_or_flex is not None and module.__class__.module_type[1] != fixed_or_flex:
                    raise ValueError('Subcontainer must only contain fixed or flex modules, not both.'
                                     f'Module {name} of type {module.__class__.module_type[1]} conflicts with previous modules'
                                     f'of type {fixed_or_flex}')
                if source_or_sink is not None and _get_source_sink(module) != source_or_sink:
                    raise ValueError('Subcontainer must only one of sources, sinks, or sources and sinks, but not combinations.'
                                     f'Module {name} of type {_get_source_sink(module)} conflicts with previous modules'
                                     f'of type {source_or_sink}')

                fixed_or_flex = module.__class__.module_type[1]
                source_or_sink = _get_source_sink(module)

        return fixed_or_flex, source_or_sink

    def module_list(self):
        l = []
        for _, values in self.data.items():
            l.extend(values)
        return l

    def iterlist(self):
        for module in self.module_list():
            yield module

    def iterdict(self):
        for name, modules in self.items():
            yield name, modules


    def __len__(self):
        return sum([len(v) for k, v in self.items()])

    def __getattr__(self, item):
        if item == 'data':
            raise AttributeError(item)
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)


class ModuleList(UserList):
    def item(self):
        if len(self) != 1:
            raise ValueError("Can only convert a ModuleList of length one to a scalar")
        return self[0]