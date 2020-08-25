function sim!(P, C, dt)
    for p in P
        integrate!(p, p.param, SNNFloat(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        record!(c)
    end
end

function sim!(P, C; dt = 0.1ms, duration = 10ms)
    @show("the top function is called a lot by this one")
    @show(P[1].param)
    @show(P)
    @show(C)
    for t = 0ms:dt:duration
        sim!(P, C, dt)
    end
end

function train!(P, C, dt, t = 0)
    for p in P
        integrate!(p, p.param, SNNFloat(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        plasticity!(c, c.param, SNNFloat(dt), SNNFloat(t))
        record!(c)
    end
end

function train!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:duration
        train!(P, C, SNNFloat(dt), SNNFloat(t))
    end
end
