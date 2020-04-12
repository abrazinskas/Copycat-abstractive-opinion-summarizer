from .constants import SPECIAL_TOKENS

try:
    import re2 as re
except ImportError:
    import re


def twitter_sentiment_token_matching(token):
    """Special token matching function for twitter sentiment data."""
    if 'URL_TOKEN' in SPECIAL_TOKENS and re.match(r'https?:\/\/[^\s]+', token):
        return SPECIAL_TOKENS['URL_TOKEN']
    if 'POS_EM_TOKEN' in SPECIAL_TOKENS and re.match(r':-?(\)|D|p)+', token):
        return SPECIAL_TOKENS['POS_EM_TOKEN']
    if 'NEG_EM_TOKEN' in SPECIAL_TOKENS and re.match(r':-?(\(|\\|/)+', token):
        return SPECIAL_TOKENS['NEG_EM_TOKEN']
    if 'USER_TOKEN' in SPECIAL_TOKENS and re.match(
            r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', token):
        return SPECIAL_TOKENS['USER_TOKEN']
    if 'HEART_TOKEN' in SPECIAL_TOKENS and re.match(r'<3+', token):
        return SPECIAL_TOKENS['HEART_TOKEN']
