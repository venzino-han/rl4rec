
# 프롬프트 템플릿 정의
PROMPT_TEMPLATES = {
    'seq_rec': {
    'title': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'task': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
    },
    'preference': {
        'title': '# User Purchase History',
        'task': '# Task\nBased on this user\'s purchase history, describe user\'s preference:',
    },
    'next_item': {
        'title': '# User Purchase History',
        'task': '# Task\nBased on this user\'s purchase history, predict what item the user will purchase next:',
    },
    'recommendation': {
        'title': '# User Purchase History',
        'task': '# Task\nBased on this user\'s purchase history, recommend suitable items for the user:',
    },
    'user_profile': {
        'title': '# User Purchase History',
        'task': '# Task\nBased on this user\'s purchase history, create a detailed user profile describing their interests and preferences:',
    },
    'recent_preference': {
        'title': '# User Purchase History',
        'task': '# Task\nBased on this user\'s purchase history, describe user\'s most recent preference:',
    },
    'reasoning': {
        'title': '# User Purchase History',
        'task': '# Task\nBased on this user\'s purchase history, reason about user\'s preference:',
    },
}
