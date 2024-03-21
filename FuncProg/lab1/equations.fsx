type FloatFunction = float -> float

let rec MachineEpsilon (epsilon: float) =
    if 1.0 + epsilon = 1.0 then epsilon*1000.0
    else MachineEpsilon (epsilon / 2.0)


let DichotomySolver (f: FloatFunction, a: float, b: float): float =
    let eps: float = MachineEpsilon(1.0)

    let rec helper (f: FloatFunction, a: float, b:float): float =
        let mid: float = (a+b)/2.0
        if abs (b - a) < eps then mid
        else
            if (f a) * (f mid) < 0.0 then helper(f, a, mid)
            else helper(f, mid, b)

    helper(f, a, b)

let IterationSolver(phi: FloatFunction, x0: float): float =
    let eps: float = MachineEpsilon(1.0)
    
    let rec helper(phi: FloatFunction, x0: float) =
        let x1: float = phi x0
        if abs (x1 - x0) < eps then x1
        else
            helper (phi, x1)

    helper (phi, x0)

let NewtonSolver (f: FloatFunction, f': FloatFunction, x0 : float) =
    let phi x = x - (f x / f' x)
    IterationSolver(phi, x0)

let f1 (x: float): float = 1.0 - x + sin(x) - log(1.0+x) 
let f2 (x: float): float = 3.0*x - 14.0 + exp(x) - exp(-x)
let f3 (x: float): float = sqrt(1.0 - x) - tan(x)

let phi1 (x: float): float = 1.0 + sin(x) - log(1.0+x)
let phi2 (x: float): float = x - 0.0432*(3.0*x - 14.0 + exp(x) - exp(-x))
let phi3 (x: float): float = atan(sqrt(1.0-x))

let f1' (x: float): float = -1.0 + cos(x) - (1.0/(1.0 + x))
let f2' (x: float): float = 3.0 + exp(x) + exp(-x)
let f3' (x: float): float = -1.0/cos(x)**2.0 - 1.0/(2.0*sqrt(1.0-x))

let main: unit =
    printfn "\t Dichotomy \t Iterations \t Newton"
    printfn "f1\t %.5f \t %.5f \t %.5f" (DichotomySolver(f1, 1.0, 1.5)) (IterationSolver(phi1, 1.25)) (NewtonSolver(f1, f1', 1.25))
    printfn "f2\t %.5f \t %.5f \t %.5f" (DichotomySolver(f2, 1.0, 3.0)) (IterationSolver(phi2, 2.0)) (NewtonSolver(f2, f2', 2.0))
    printfn "f3\t %.5f \t %.5f \t %.5f" (DichotomySolver(f3, 0.0, 1.0)) (IterationSolver(phi3, 0.5)) (NewtonSolver(f3, f3', 0.5))
