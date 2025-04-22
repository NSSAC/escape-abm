"""Type inference / checking."""

from contextlib import contextmanager
from typing import assert_never

import networkx as nx
from typeguard import TypeCheckError, check_type

from . import ast
from . import types
from .misc import CodeError, SourcePosition, Scope


@contextmanager
def err_desc(description: str, pos: SourcePosition | None):
    try:
        yield
    except AssertionError:
        raise CodeError("Type Error", description, pos)


def is_lvalue_ref(ref: ast.Reference) -> bool:
    try:
        check_type(ref.value, ast.LValueRef)
        return True
    except TypeCheckError:
        return False


def is_callable_ref(ref: ast.Reference) -> bool:
    try:
        check_type(ref.value, ast.CValueRef)
        return True
    except TypeCheckError:
        return False


def is_rvalue_expr(expr: ast.Expression) -> bool:
    match expr:
        case ast.Literal():
            return True
        case ast.Reference() as ref:
            try:
                check_type(ref.value, ast.DeviceRValueRef)
                return True
            except TypeCheckError:
                return False
        case ast.UnaryExpression() as uexpr:
            return is_rvalue_expr(uexpr)
        case ast.BinaryExpression() as bexpr:
            return is_rvalue_expr(bexpr.left) and is_rvalue_expr(bexpr.right)
        case ast.ParenthesizedExpression() as pexpr:
            return is_rvalue_expr(pexpr.expression)
        case ast.FunctionCall() as fcall:
            if not is_callable_ref(fcall.function):
                return False
            for arg in fcall.args:
                if not is_rvalue_expr(arg):
                    return False
            return True
        case _ as unexpected:
            assert_never(unexpected)


def ref_type(ref: ast.Reference) -> types.Type:
    value = ref.value
    try:
        check_type(value, ast.AllRef)
    except TypeCheckError:
        raise CodeError("Reference error", "Invalid reference", ref.pos)

    if isinstance(value, tuple):
        value = value[-1]

    if isinstance(value, types.Type):
        return value
    elif isinstance(value, types.FunctionType):
        return value.return_
    else:
        raise CodeError("Type error", "Unable to infer type of reference", ref.pos)


def infer_type(x: ast.Expression, type_tree: nx.DiGraph) -> str:
    match x:
        case ast.Literal() as lit:
            match lit.value:
                case bool():
                    return "bool"
                case int():
                    return "int"
                case float():
                    return "float"
                case str():
                    return "str"
        case ast.Reference() as ref:
            return ref_type(ref).name
        case ast.UnaryExpression() as uexpr:
            type = infer_type(uexpr.argument, type_tree)
            if uexpr.operator in ast.UnaryBooleanOperators:
                with err_desc(
                    "Unary boolean operator used with non boolean expression",
                    uexpr.pos,
                ):
                    assert type == "bool"
                return type
            elif uexpr.operator in ast.UnaryNumericOperators:
                with err_desc(
                    "Unary numeric operator used with non numeric expression",
                    uexpr.pos,
                ):
                    assert types.is_sub_type(type, "float", type_tree)
                return type
            else:
                raise RuntimeError(f"Unepxected unary operator {uexpr.operator}")
        case ast.BinaryExpression() as bexpr:
            type1 = infer_type(bexpr.left, type_tree)
            type2 = infer_type(bexpr.right, type_tree)
            if bexpr.operator in ast.BinaryBooleanOperators:
                with err_desc(
                    "Binary boolean operator used with non boolean expression",
                    bexpr.pos,
                ):
                    assert type1 == "bool"
                    assert type2 == "bool"
                return type1
            elif bexpr.operator in ast.BinaryNumericOperators:
                with err_desc(
                    "Binary numeric operator used with non numeric expression",
                    bexpr.pos,
                ):
                    assert types.is_sub_type(type1, "float", type_tree)
                    assert types.is_sub_type(type2, "float", type_tree)
                return types.lub_type(type1, type2, type_tree)
            elif bexpr.operator in ast.BinaryIntegerOperators:
                with err_desc(
                    "Binary integer operator used with non integer expression",
                    bexpr.pos,
                ):
                    assert type1 in ("int", "uint")
                    assert type2 in ("int", "uint")
                return types.lub_type(type1, type2, type_tree)
            elif bexpr.operator in ast.BinaryEqualityOperators:
                if type1 == type2:
                    return "bool"

                with err_desc("Can't compare equality; type mismatch", bexpr.pos):
                    assert types.is_sub_type(type1, "float", type_tree)
                    assert types.is_sub_type(type2, "float", type_tree)

                return "bool"
            elif bexpr.operator in ast.BinaryOrderOperators:
                if type1 == type2 and types.is_sub_type(type1, "_enum", type_tree):
                    return "bool"

                with err_desc("Can't compare; type mismatch", bexpr.pos):
                    assert types.is_sub_type(type1, "float", type_tree)
                    assert types.is_sub_type(type2, "float", type_tree)

                return "bool"
            else:
                raise RuntimeError(f"Unepxected binary operator {bexpr.operator}")
        case ast.ParenthesizedExpression() as pexpr:
            return infer_type(pexpr.expression, type_tree)
        case ast.FunctionCall() as fn_call:
            with err_desc("Expected a callable reference", fn_call.pos):
                assert is_callable_ref(fn_call.function)

            fn_type: ast.FunctionType = fn_call.function.value.type

            with err_desc("Invalid argument count", fn_call.pos):
                assert len(fn_type.params) == len(fn_call.args)

            for idx, (ptype, arg) in enumerate(zip(fn_type.params, fn_call.args), 1):
                with err_desc(f"Invalid argument type for arg {idx}", fn_call.pos):
                    atype = infer_type(arg, type_tree)
                    assert types.is_sub_type(atype, ptype.name, type_tree)

            return fn_type.return_.name
        case _ as unexpected:
            assert_never(unexpected)


def type_check(x: ast.TypeCheckable, type_tree: nx.DiGraph):
    match x:
        case ast.GlobalVariable() as gvar:
            var_type = gvar.type.name
            def_type = infer_type(gvar.default, type_tree)
            with err_desc(
                "Expression of type {type1.name} can't be assigned to variable of type {type2.name}",
                gvar.pos,
            ):
                assert types.is_sub_type(def_type, var_type, type_tree)
        case ast.NormalDist() as ndist:
            with err_desc("Expected numeric expression", ndist.mean.pos):
                assert types.is_sub_type(
                    infer_type(ndist.mean, type_tree), "float", type_tree
                )
            with err_desc("Expected numeric expression", ndist.std.pos):
                assert types.is_sub_type(
                    infer_type(ndist.std, type_tree), "float", type_tree
                )
            with err_desc("Expected numeric expression", ndist.min.pos):
                assert types.is_sub_type(
                    infer_type(ndist.min, type_tree), "float", type_tree
                )
            with err_desc("Expected numeric expression", ndist.max.pos):
                assert types.is_sub_type(
                    infer_type(ndist.max, type_tree), "float", type_tree
                )
        case ast.UniformDist() as udist:
            with err_desc("Expected numeric expression", udist.low.pos):
                assert types.is_sub_type(
                    infer_type(udist.low, type_tree), "float", type_tree
                )
            with err_desc("Expected numeric expression", udist.high.pos):
                assert types.is_sub_type(
                    infer_type(udist.high, type_tree), "float", type_tree
                )
        case ast.DiscreteDist() as ddist:
            for p in ddist.ps:
                with err_desc("Expected numeric expression", p.pos):
                    assert types.is_sub_type(
                        infer_type(p, type_tree), "float", type_tree
                    )
            for v in ddist.vs:
                with err_desc("Expected numeric expression", v.pos):
                    assert types.is_sub_type(
                        infer_type(v, type_tree), "float", type_tree
                    )
        case ast.Contagion():
            pass
        case ast.Function() | ast.LambdaFunction() as fn:
            for stmt in fn.body:
                type_check(stmt, type_tree)
        case ast.PassStatement():
            pass
        case ast.CallStatement() as call:
            infer_type(call.call, type_tree)
        case ast.ReturnStatement() as ret:
            expected_type = ret.function.type.return_.name
            expr_type = infer_type(ret.expression, type_tree)
            with err_desc("Return type mismatch", ret.pos):
                assert types.is_sub_type(expr_type, expected_type, type_tree)
        case ast.IfStatement() as ifs:
            expected_type = "float"
            expr_type = infer_type(ifs.condition, type_tree)
            with err_desc("Invalid condition", ifs.condition.pos):
                assert types.is_sub_type(expr_type, expected_type, type_tree)
            for stmt in ifs.body:
                type_check(stmt, type_tree)
            for elif_ in ifs.elifs:
                type_check(elif_, type_tree)
            if ifs.else_ is not None:
                type_check(ifs.else_, type_tree)
        case ast.ElifSection() as elif_:
            expected_type = "float"
            expr_type = infer_type(elif_.condition, type_tree)
            with err_desc("Invalid condition", elif_.condition.pos):
                assert types.is_sub_type(expr_type, expected_type, type_tree)
            for stmt in elif_.body:
                type_check(stmt, type_tree)
        case ast.ElseSection() as else_:
            for stmt in else_.body:
                type_check(stmt, type_tree)
        case ast.SwitchStatement() as sws:
            a_type = infer_type(sws.condition, type_tree)
            for case_ in sws.cases:
                b_type = infer_type(case_.match, type_tree)
                if a_type != b_type:
                    with err_desc(
                        "Can't compare with switch condition", case_.match.pos
                    ):
                        assert types.is_sub_type(a_type, "float", type_tree)
                        assert types.is_sub_type(b_type, "float", type_tree)

                for stmt in case_.body:
                    type_check(stmt, type_tree)

            if sws.default is not None:
                for stmt in sws.default.body:
                    type_check(stmt, type_tree)
        case ast.WhileLoop() as loop:
            expected_type = "float"
            expr_type = infer_type(loop.condition, type_tree)
            with err_desc("Invalid condition", loop.condition.pos):
                assert types.is_sub_type(expr_type, expected_type, type_tree)
            for stmt in loop.body:
                type_check(stmt, type_tree)
        case ast.AssignmentStatement() as astmt:
            with err_desc("Not a valid assignable value", astmt.lvalue.pos):
                assert is_lvalue_ref(astmt.lvalue)

            with err_desc("Can't evaluate expression", astmt.rvalue.pos):
                assert is_rvalue_expr(astmt.rvalue)

            lvalue_type = infer_type(astmt.lvalue, type_tree)
            rvalue_type = infer_type(astmt.rvalue, type_tree)
            with err_desc(
                f"Can't assign expression of type {rvalue_type} to variable of type {lvalue_type}",
                astmt.pos,
            ):
                assert types.is_sub_type(rvalue_type, lvalue_type, type_tree)
        case ast.UpdateStatement() as ustmt:
            with err_desc("Not a valid assignable value", ustmt.lvalue.pos):
                assert is_lvalue_ref(ustmt.lvalue)

            with err_desc("Can't evaluate expression", ustmt.rvalue.pos):
                assert is_rvalue_expr(ustmt.rvalue)

            lvalue_type = infer_type(ustmt.lvalue, type_tree)
            rvalue_type = infer_type(ustmt.rvalue, type_tree)
            if ustmt.operator in ast.UpdateNumericOperators:
                with err_desc("Invalid numeric update", ustmt.pos):
                    assert types.is_sub_type(lvalue_type, "float", type_tree)
                    assert types.is_sub_type(rvalue_type, "float", type_tree)
            elif ustmt.operator in ast.UpdateIntegeOperators:
                with err_desc("Invalid integer update", ustmt.pos):
                    assert types.is_sub_type(lvalue_type, "int", type_tree)
                    assert types.is_sub_type(rvalue_type, "int", type_tree)
            else:
                raise RuntimeError(f"Unepxected update operator {ustmt.operator}")
        case ast.ParallelStatement() as pstmt:
            pass
