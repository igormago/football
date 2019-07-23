import sys


def get_features(config):

    method_to_call = getattr(sys.modules[__name__], str(config.features))
    return method_to_call(config)


def accum_goals(config):

    features_trends = ['goals']
    locales = ['home', 'away']
    features = list()

    if not config.is_minute_feature:
        for t in features_trends:
            for l in locales:
                col = '_'.join(['accum_trends', t, l, str(config.minute)])
                features.append(col)
    else:
        for t in features_trends:
            for l in locales:
                col = '_'.join(['accum_trends', t, l])
                features.append(col)

        features.append('minute')

    return features


def accum_features(config):

    features_trends = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
    features_cards = ['yellow_cards', 'red_cards']
    features_ratio = ['possession']
    locales = ['home', 'away']
    features = list()

    if not config.is_minute_feature:
        for t in features_trends:
            for l in locales:
                col = '_'.join(['accum_trends', t, l, str(config.minute)])
                features.append(col)

        for t in features_cards:
            for l in locales:
                col = '_'.join(['accum_cards', t, l, str(config.minute)])
                features.append(col)

        for t in features_ratio:
            col = '_'.join(['ratio_trends', t, str(config.minute)])
            features.append(col)

    else:
        for t in features_trends:
            for l in locales:
                col = '_'.join(['accum_trends', t, l])
                features.append(col)

        for t in features_cards:
            for l in locales:
                col = '_'.join(['accum_cards', t, l])
                features.append(col)

        for t in features_ratio:
            col = '_'.join(['ratio_trends', t])
            features.append(col)

        features.append('minute')

    return features
