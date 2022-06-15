def get_jobname_to_id(name):
    dict = {
        'bagboost' : '4082125',
        'fastbag_ham' : '4083560',
        'ORMOGP' : '4082885'
    }

    if name in dict.keys():
        return dict[name]
    return name