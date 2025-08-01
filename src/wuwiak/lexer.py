import parsy as P

keywords = [
    'dopóty dopóki',
    'if',
    'else',
    'return',
    'true',
    'false',
]

whitespaces = P.regex(r'\s+', P.re.MULTILINE).desc("whitespace")
linecomment = P.regex(r'#.*\n').desc("line comment")
skip = P.alt(
    whitespaces,
    linecomment,
).many()


def lex(lexeme):
    """
    Single lexeme proceeded by a skip
    """
    return lexeme << skip

def token(s):
    """
    Fixed token
    """
    return lex(P.string(s))


def rtoken(r):
    """
    Fixed token (by regexp)
    """
    return lex(P.regex(r))


integer = rtoken(r'-?\d+').desc("integer").map(int)

lparen = token('(')
rparen = token(')')

lbrac = token('{')
rbrac = token('}')

lsbrac = token('[')
rsbrac = token(']')


def kw(kw):
    if kw not in keywords:
        raise ValueError(f"Not a keyword: {kw}")

    return token(kw)


@P.generate
def gen_ident():
    name = yield P.regex(r'[a-zA-Z_][a-zA-Z0-9_]*')
    if name in keywords:
        return P.fail("invalid identifier")
    return name


ident = lex(gen_ident).desc("identifier")


@P.generate
def gen_string():
    yield P.string('"')
    chars = []

    while True:
        c = yield P.alt(
            P.regex(r'[^"\\]'),  # Any char except escape and close
            P.string('\\') >>  # Escape sequence and...
            P.alt(
                P.string('"'),
                P.string('\\'),
                P.string('n').result('\n'),
                P.string('t').result('\t'),
                P.string('r').result('\r'),
                P.string('b').result('\b'),
            )
        ).optional()

        if c is None:
            break

        chars.append(c)

    yield P.string('"')
    return ''.join(chars)


string = lex(gen_string).desc("string literal")

op = lex(P.alt(
    P.string('+'),
    P.string('-'),
    P.string('*'),
    P.string('/'),
    P.string('%'),
    P.string('>='),
    P.string('<='),
    P.string('=='),
    P.string('!='),
    P.string('>'),
    P.string('<'),
    P.string('||'),
    P.string('&&'),
).desc("operator"))

unop = lex(P.alt(
    P.string('!')
).desc("operator"))

semicolon = token(';')
