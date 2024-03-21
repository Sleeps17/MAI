let rec MachineEpsilon (epsilon: float) =
    if 1.0 + epsilon = 1.0 then epsilon
    else MachineEpsilon (epsilon / 2.0)

let Ln(x: float) : float = log (1.0 + x - 2.0*x*x)

let rec Power(num: float, exponent: int): float =
    if exponent = 0 then 1.0
    else num * Power(num, (exponent - 1))

let Coefficient(n: int): float =
    (Power(-1.0, n+1)*Power(2.0, n) - 1.0) / (float n)

let DumbTeylor (x: float): (float*int) =
    let eps: float = MachineEpsilon(1.0)
    let next(x: float, n: int): float =
        Coefficient(n)*Power(x, n)

    let rec helper n curr =
        let elem: float = next(x, n)

        if abs(elem) < eps then (curr, n)
        else helper (n+1) (curr+elem)

    helper 1 0.0



let SmartTeylor(x: float): (float*int) =
    let eps: float = MachineEpsilon(1.0)
    let k(x: float, n: int): float =
        x*Coefficient(n) / Coefficient(n-1)
        
    let next(prev: float, x: float, n: int) =
        if n = 1 then x
        else k(x, n)*prev

    let rec helper n curr prev =
        let elem: float = next(prev, x, n)

        if abs(elem) < eps then (curr, n)
        else helper (n+1) (curr + elem) (elem)
    
    helper 1 0.0 0.0

let table(a: float, b: float, steps: int): unit =
    printfn "  x \t     Builtin    Smart Taylor  #terms   Dumb Taylor    #terms"
    for i=0 to steps do
        let x = a + (float i)/(float steps)*(b-a)
        let (r1, k1) = DumbTeylor x
        let (r2, k2) = SmartTeylor x
        printfn "%10.6f %10.6f   %10.6f\t %d   %10.6f\t %d" x (Ln x) r1 k1 r2 k2 

let main: unit =
    let a: float = -0.2
    let b: float = 0.3

    table(a, b, 20)