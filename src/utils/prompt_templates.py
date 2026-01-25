
# 프롬프트 템플릿 정의
PROMPT_TEMPLATES = {
    'seq_rec': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
    },
    
    'seq_rec_date': {
    'head': '### Role\n' +\
            'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
            'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n\n' +\
            '### Task\n' +\
            'Determine the **"Current Interest Session Start Date"** relative to the Target Purchase Date.\n' +\
            'Analyze the user\'s history to find the point where the current shopping intent began.\n' +\
            'Consider both **Semantic Relevance** (is it related?) and **Temporal Proximity** (is it recent enough to matter?).\n' +\
            '\n### Input:\n',

    'tail': '### Instructions\n' +\
            '1. Calculate the time gap between items and the Target Date.\n' +\
            '2. Identify any significant "Time Breaks" (e.g., > 1 month of inactivity) that suggest a shift in intent.\n' +\
            '3. Determine the **Cut-off Date**. Items before this date are considered "Past History", and items after are "Current Context".\n' +\
            '4. Based on the user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
    },

    'seq_rec_recent': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.\n' +\
            'The response should be start with "Considering the user\'s most recent purchase of [Last Item Name], ..."',
    },

    'preference': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, describe user\'s preference:',
    },
    'next_item': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, predict what item the user will purchase next:',
    },
    'recommendation': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, recommend suitable items for the user:',
    },
    'user_profile': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, create a detailed user profile describing their interests and preferences:',
    },
    'recent_preference': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, describe user\'s most recent preference:',
    },
    'reasoning': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, reason about user\'s preference:',
    },
    'feature_reasoning_rec': {
        'head': 'You are an expert product analyst and shopping assistant. Below is a list of items a user has purchased recently.\n' +\
                 'Your task is to deeply analyze the underlying product features (such as brand, material, style, and functionality) that drive this user\'s choices.\n' +\
                 'Instead of looking at items in isolation, identify the common attributes that connect these purchases to infer the user\'s specific tastes.\n' +\
                 'Below is the user purchase history:\n',
        'tail': '# Task\n' +\
                'First, analyze and summarize the user\'s key preferences regarding specific product features (e.g., brands, materials, functionalities, or styles) derived from the history.\n' +\
                'Then, based on this feature summary, recommend suitable items that align with the user\'s taste:',
    },
    'feature_reasoning_recent_rec': {
        'head': 'You are an expert product analyst and shopping assistant. Below is a list of items a user has purchased recently.\n' +\
                 'Your task is to deeply analyze the underlying product features (such as brand, material, style, and functionality) that drive this user\'s choices.\n' +\
                 'Instead of looking at items in isolation, identify the common attributes that connect these purchases to infer the user\'s specific tastes, **giving more weight to recent interactions**.\n' +\
                 'Below is the user purchase history **(ordered by time)**:\n',
        'tail': '# Task\n' +\
                'First, analyze and summarize the user\'s key preferences regarding specific product features (e.g., brands, materials, functionalities, or styles) derived from the history, **focusing on their latest interests**.\n' +\
                'Then, based on this feature summary, recommend suitable items that align with the user\'s **current** taste:',
    },
    
    'seq_rec_with_sasrec': {
        'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
        'sasrec_section': '\n# SASRec Recommender Results\n' +\
                          'A collaborative filtering recommender (SASRec) has suggested the following items based on user behavior patterns. ' +\
                          '**Note: These recommendations are not always accurate and should be used as reference only** when generating your query terms:\n',
        'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products. ' +\
                'You may refer to the SASRec recommendations above, but they are not always correct, so use your own judgment based on the purchase history.',
    },
    
    'seq_rec_recent_with_sasrec': {
        'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
        'sasrec_section': '\n# SASRec Recommender Results\n' +\
                          'A collaborative filtering recommender (SASRec) has suggested the following items based on user behavior patterns. ' +\
                          '**Note: These recommendations are not always accurate and should be used as reference only** when generating your query terms:\n',
        'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products. ' +\
                'The response should start with "Considering the user\'s most recent purchase of [Last Item Name], ..."\n' +\
                'You may refer to the SASRec recommendations above, but they are not always correct, so use your own judgment based on the purchase history.',
    },

}
