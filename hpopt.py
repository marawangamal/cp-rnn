from vizier.service import clients
from vizier.service import pyvizier as vz


def hpopt(hps, hp_ranges, hp_types, evaluate_fn, evaluate_kwargs=dict(), iters=10, metric_name="bpc"):
    # Algorithm, search space, and metrics.
    study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT')

    add_param = {
        "int": study_config.search_space.root.add_int_param,
        "float": study_config.search_space.root.add_float_param,
        "discrete": study_config.search_space.root.add_discrete_param
    }

    for hp, hp_range, hp_type in zip(hps, hp_ranges, hp_types):
        study_config.search_space.root.add_float_param('w', 0.0, 5.0)
        add_param[hp_type](hp, *hp_range)

    study_config.metric_information.append(vz.MetricInformation('metric_name', goal=vz.ObjectiveMetricGoal.MINIMIZE))

    # Setup client and begin optimization. Vizier Service will be implicitly created.
    study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')
    for i in range(iters):
        suggestions = study.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            # objective = evaluate_fn(params['w'], params['x'], params['y'], params['z'], **evaluate_kwargs)
            objective = evaluate_fn(**params, **evaluate_kwargs)
            suggestion.complete(vz.Measurement({metric_name: objective}))




if __name__ == '__main__':
    hpopt()
