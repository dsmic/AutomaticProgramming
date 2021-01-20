# this only seperates names from other tokens, as a minimal parser which might work for different programming languages

#import sys
#import traceback
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


cuda = torch.device('cuda')

name_chars = set('0123456789abcdefghijklmnopyrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def is_name_char(c):
    if c in name_chars:
        return True
    else:
        return False
    
#ff = open(sys.argv[1])

def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]

class func_provider():
    def __init__(self, filenames, epochs=1):
        self.dataset = read_file(filenames)
        self.epochs = epochs
        pass
    def generator(self):
#        func_list = []
        self.cc = 0
        for _ in range(self.epochs):
            for fn in self.dataset:
                self.cc += 1
                ff = open(fn)
                function_lines = []
                try:
                    multi_line_comment_in_progress = False
                    for line in ff.readlines(): # sys.stdin:
                        #print(line)
                        is_comment = True
                        #print('*' + line)
                        pp = 0
                        stop_multiline = False
                        for c in line:
                            if not c.isspace():
                                if multi_line_comment_in_progress:
                                    if line.find('"""') >= 0:
                                        stop_multiline = True
                                else:
                                    if line[pp:pp+3] == '"""':
                                        multi_line_comment_in_progress = True
                                        if line[pp+3:].find('"""') >= 0:
                                            stop_multiline = True
                                if c != '#':
                                    is_comment = False
                                break
                            else:
                                pp += 1
                                
                        if not is_comment and not multi_line_comment_in_progress:
                            #print('-'+ line)
                            if line[:3] == 'def': # or line[:5] == 'class':
                                if len(function_lines) > 0:
                                    #print('***\n' + function_string + '====\n') # last function
                                    yield function_lines
                                    #func_list.append(function_string)
                                function_lines = [line]
                            else:
                                if len(line) > 0 and len(function_lines) > 0:
                                    if line[0].isspace():
                                        function_lines.append(line)
                                    else:
                                        if len(function_lines) > 0:
                                            #print('***\n' + function_string + '===\n')
                                            yield function_lines
                                            #func_list.append(function_string)
                                        function_lines = []
                        if stop_multiline:
                            multi_line_comment_in_progress = False
                except UnicodeDecodeError as e:
                    print(fn + '\n  wrong encoding ' + str(e))
        

import ast
import textwrap

def _parse_assign(node, scope):
    value = _to_str(node.value, scope)
    target_iter = _to_str_iter(node.targets, scope)

    assignments = []

    for t in target_iter:
        is_in_scope = t in scope
        is_field_assignment = '.' in t
        is_arr_assignment = '[' in t

        if is_in_scope or is_field_assignment or is_arr_assignment:
            assignments.append(f"{t} = {value}")
        else:
            scope.add(t)
            assignments.append(f"var {t} = {value}")

    return ";".join(assignments)


def _parse_bool_op(node, scope):
    op = _to_str(node.op, scope)
    return op.join(_to_str_iter(node.values, scope))


def _parse_compare(node, scope):
    ops = _to_str_iter(node.ops, scope)
    comparators = _to_str_iter(node.comparators, scope)
    ops_comps = zip(ops, comparators)
    return "%s %s" % (
        _to_str(node.left, scope),
        " ".join("%s %s" % oc for oc in ops_comps),
    )


def _parse_call(node, scope):
    func = _to_str(node.func, scope)
    args = _to_str_iter(node.args, scope)
    return "%s(%s)" % (
        func,
        ", ".join(args),
    )


def _parse_dict(node, scope):
    keys = _to_str_iter(node.keys, scope)
    values = _to_str_iter(node.values, scope)
    kvs = zip(keys, values)
    return "{%s}" % ", ".join("%s: %s" % kv for kv in kvs)


def _parse_function_def(node, scope):
    new_scope = scope.enter_new()
    new_scope.add(x.arg for x in node.args.args)

    return "function %(name)s(%(args)s) {\n%(body)s\n}" % {
        "name": node.name,
        "args": _to_str(node.args, new_scope),
        "body": _to_str(node.body, new_scope),
    }


def _parse_lambda(node, scope):
    new_scope = scope.enter_new()
    new_scope.add(x.arg for x in node.args.args)

    return "((%(args)s) => (%(body)s))" % {
        "args": _to_str(node.args, new_scope),
        "body": _to_str(node.body, new_scope),
    }


def _parse_list(node, scope):
    return "[%s]" % ", ".join(_to_str(x, scope) for x in node.elts)


def _parse_arguments(node, scope):
    return ", ".join(x.arg for x in node.args)


# See:
# - https://docs.python.org/3/library/ast.html
# - https://greentreesnakes.readthedocs.io/en/latest/nodes.html
_PARSERS = {
    #"Module":
    "FunctionDef": _parse_function_def,
    #"AsyncFunctionDef":
    #"ClassDef": _parse_class_def,  # TODO: Need to figure out "new" JS keyword.
    "Return": "return %(value)s",
    "Delete": "delete %(targets)s",
    "Assign": _parse_assign,
    "AugAssign": "%(target)s %(op)s= %(value)s",
    #"AnnAssign":
    "For": "%(iter)s.forEach((%(target)s, _i) => {\n%(body)s\n})",
    #"AsyncFor":
    "While": "while (%(test)s) {\n%(body)s\n}",
    "If": "if (%(test)s) {\n%(body)s\n} else {\n%(orelse)s\n}",
    #"With":
    #"AsyncWith":
    "Raise": "throw new Error(%(exc)s)",
    #"Try": TODO _parse_try,
    #"TryFinally": TODO _parse_try_finally,
    #"TryExcept": TODO _parse_try_except,
    #"Assert":
    #"Import":
    #"ImportFrom":
    #"Global":
    #"Nonlocal":
    "Expr": "%(value)s",
    "Pass": "",
    "BoolOp": _parse_bool_op,
    #"NamedExpr":
    "BinOp": "(%(left)s %(op)s %(right)s)",
    "UnaryOp": "(%(op)s%(operand)s)",
    "Lambda": _parse_lambda,
    "IfExp": "(%(test)s) ? (%(body)s) : (%(orelse)s)",
    "Dict": _parse_dict,
    #"Set":
    #"ListComp":
    #"SetComp":
    #"DictComp":
    #"GeneratorExp":
    #"Await":
    #"Yield":
    #"YieldFrom":
    "Compare": _parse_compare,
    "Call": _parse_call,
    #"FormattedValue":
    #"JoinedStr":
    "Constant": "%(value)s",
    "Attribute": "%(value)s.%(raw_attr)s",
    "Subscript": "%(value)s[%(slice)s]",
    #"Starred":
    "Name": "%(raw_id)s",
    "List": _parse_list,
    #"Tuple": TODO
    #"AugLoad":
    #"AugStore":
    #"Param":
    #"Slice":
    #"ExtSlice":
    "Index": "%(value)s",
    "And": "&&",
    "Or": "||",
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    #"MatMult":
    "Div": "/",
    "Mod": "%%",  # Escape the "%" as "%%" since we call "%" on this string later.
    #"Pow":
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitXor": "^",
    "BitAnd": "&",
    #"FloorDiv": ,
    "Invert": "~",
    "Not": "!",
    "UAdd": "+",
    "USub": "-",
    "Eq": "===",
    "NotEq": "!==",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    #"Is":
    #"IsNot":
    #"In":
    #"NotIn":
    #"ExceptHandler": _parse_except_handler,
    "Break": "break",
    "Continue": "continue",
    "arguments": _parse_arguments,

    # For Python < 3.8
    "Num": "%(n)s",
    "Str": '%(s)s',
    "Bytes": '"%(s)s"',
    #"Ellipsis
    "NameConstant": "%(value)s",
}


def _to_str(node, scope):
    node_type = type(node)

    if node_type is list:
        return ";\n".join(_to_str(x, scope) for x in node)

    if node is None:
        return "null"

    if node_type is str:
        return '"%s"' % node

    if node_type in (int, float):
        return str(node)

    if node_type is bool:
        return "true" if node else "false"

    if node_type.__name__ not in _PARSERS:
        raise Exception("Unsupported operation in JS: %s" % node_type)

    parser = _PARSERS[node_type.__name__]

    if type(parser) is str:
        return parser % _DictWrapper(node.__dict__, scope)

    return parser(node, scope)


class _DictWrapper(dict):
    def __init__(self, dikt, scope):
        self._dict = dikt
        self._parsed_keys = set()
        self._scope = scope

    def __getitem__(self, k):
        raw = False

        if k.startswith("raw_"):
            k = k[4:]
            raw = True

        if k not in self._parsed_keys:
            if raw:
                self._dict[k] = self._dict[k]
            else:
                self._dict[k] = _to_str(self._dict[k], self._scope)
            self._parsed_keys.add(k)

        return self._dict[k]


def _to_str_iter(arg, scope):
    return (_to_str(x, scope) for x in arg)


class _Scope(object):
    def __init__(self, parent=None):
        self._parent = parent
        self._identifiers = set()

    def enter_new(self):
        return _Scope(self)

    def add(self, identifiers):
        for x in identifiers:
            self._identifiers.add(x)

    def __contains__(self, x):
        if x in self._identifiers:
            return True

        if self._parent is not None and x in self._parent:
            return True

        return False

class JSFuncStr(object):
    def __init__(self, source_code, initial_scope={}):
        # self._orig = func
        self._initial_scope = initial_scope

        # source_code = inspect.getsource(func)
        code_ast = ast.parse(textwrap.dedent(source_code))
        self._code = code_ast.body[0].body

        empty_scope = _Scope()
        initial_code_py = '\n'.join(
            "%s = %s" % (k, _to_str(v, empty_scope))
            for (k, v) in self._initial_scope.items()
        )

        if initial_code_py:
            initial_code_ast = ast.parse(textwrap.dedent(initial_code_py))
            self._initial_code_js = _to_str(
                initial_code_ast.body, empty_scope) + ";"
        else:
            self._initial_code_js = ""


    def __str__(self):
        return self._initial_code_js + _to_str(
            self._code,
            _Scope(self._initial_scope)
        )

    def __call__(self, *args, **kwargs):
        return self._orig(*args, **kwargs)

 
# sss = '''
# def sum_and_check_if_42(a, b):
#     c = a + b
#     if c == 42:
#       return True
#     else:
#       return False

#     result = sum_and_check_if_42(10, 30)
#     console.log("Is it 42?", result)
# '''

# a = JSFuncStr('@js\ndef js_code():\n\n  ' + sss.replace('\n','\n  '))


def mini_tokenizer(lines):
    res = [] 
    for l in lines:   
        tmp = ''
        name = False
        for c in l:
            new_name = c in name_chars
            if name and new_name:
                tmp += c
            else:
                name = new_name
                if len(tmp) > 0: # and not tmp.isspace():
                    #print(res,tmp)
                    if len(res) > 0 and res[-1].isspace() and tmp.isspace() and res[-1] !='\n':
                        res = res[:-1] + [res[-1] + tmp] # concatenate whitespaces for python
                    else:
                        res.append(tmp)
                tmp = '' + c

    return res





functions_train = func_provider('python100k_train.txt_random', 10).generator()
functions_test = func_provider('python50k_eval.txt_random', 100).generator()

max_len = 200
max_tokens = 1000
hidden_size = max_tokens * 5
ToCategorical = torch.eye(max_tokens)


def error(pred, target): return ((pred-target)**2).mean()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(max_tokens, max_tokens)
        self.fc1 = nn.Linear(max_tokens, hidden_size)
        self.fc2 = nn.Linear(hidden_size, max_tokens)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #y, (self.hidden_state, self.cell_state) = self.lstm_layer(x, (self.hidden_state, self.cell_state)) # would be statefull, but problem with backward ??
        x = self.embedding(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

net_model = Model().to(cuda)

def token_to_int(token_list, lenret = None):
    ret = [hash(x) % max_tokens for x in token_list]
    if lenret is not None:
        if len(ret) > lenret:
            raise Exception("funtion is to long")
        return ret + [0]*(lenret - len(ret))
    return [hash(x) % max_tokens for x in token_list]


#optimizer = torch.optim.SGD(net_model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
optimizer = torch.optim.AdamW(net_model.parameters(), lr=0.00002)
ccc = 0
debug = False
float_loss = None
test_loss = None
smoothing = 0.001
for s in functions_train:
    try:
        pycode = ''.join(s)
        jscode = str(JSFuncStr('@js\ndef js_code():\n\n  ' + ''.join(s).replace('\n','\n  ')))
        pytokens = mini_tokenizer([pycode])
        jstokens = mini_tokenizer([jscode])
        
        if debug:
            print('111111111111111111111111111\n' + pycode)
            print(pytokens, token_to_int(pytokens))
            print('222222222222222222222222222\n' + jscode)
            print(jstokens, token_to_int(jstokens))
        oo = ToCategorical[token_to_int(jstokens, max_len)].reshape(1,-1,max_tokens).to(cuda)
        optimizer.zero_grad()
        otmp = net_model.forward(torch.tensor(token_to_int(jstokens, max_len)).to(cuda))
        acc = float((torch.argmax(oo,2) == torch.argmax(otmp,1)).type(torch.FloatTensor).mean())
        loss = error(oo, otmp)
        loss.backward()
        optimizer.step()
        if float_loss is None:
            float_loss = float(loss)
            float_acc = float(acc)
        else:
            float_loss = (1-smoothing) * float_loss + smoothing * float(loss)
            float_acc  = (1-smoothing) * float_acc  + smoothing * float(acc)
            
        ccc += 1
        print('{:7d} Loss = {:10.7f} {:10.7f} acc = {:10.7f} {:10.7f}'.format(ccc, float(loss), float(float_loss), float(acc), float(float_acc)))
        writer.add_scalar('Loss/train', loss, ccc)
        writer.add_scalar('Accuracy/train', acc, ccc)
        ok = True
        while ok:
            try:
                s = next(functions_test)
                pycode = ''.join(s)
                jscode = str(JSFuncStr('@js\ndef js_code():\n\n  ' + ''.join(s).replace('\n','\n  ')))
                pytokens = mini_tokenizer([pycode])
                jstokens = mini_tokenizer([jscode])
                
                if debug:
                    print('111111111111111111111111111\n' + pycode)
                    print(pytokens, token_to_int(pytokens))
                    print('222222222222222222222222222\n' + jscode)
                    print(jstokens, token_to_int(jstokens))
                oo = ToCategorical[token_to_int(jstokens, max_len)].reshape(1,-1,max_tokens).to(cuda)
                optimizer.zero_grad()
                otmp = net_model.forward(torch.tensor(token_to_int(jstokens, max_len)).to(cuda))
                acc = float((torch.argmax(oo,2) == torch.argmax(otmp,1)).type(torch.FloatTensor).mean())
                loss = error(oo, otmp)
                if test_loss is None:
                    test_loss = float(loss)
                    test_acc = float(acc)
                else:
                    test_loss = (1-smoothing) * test_loss + smoothing * float(loss)
                    test_acc  = (1-smoothing) * test_acc  + smoothing * float(acc)
                writer.add_scalar('Loss/test', loss, ccc)
                writer.add_scalar('Accuracy/test', acc, ccc)
                print('   test Loss = {:10.7f} {:10.7f} acc = {:10.7f} {:10.7f}'.format(float(loss), float(test_loss), float(acc), float(test_acc)))
                ok = False
            except Exception:
                pass
            
    except Exception as eee:
        if debug:
            print('jot able to generate javascript: '+ str(eee))#  + traceback.format_exc())
    
    

