using Debugger

function foo(n)
    x = n+1
    @bp
    return ((BigInt[1 1; 1 0])^x)[2,1]
end

@run foo(20)
