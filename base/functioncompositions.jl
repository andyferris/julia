"""
`SplattingFunctor` is a singleton type (with instance `splat`) that indicates a
composed function should splat arguments into a function, such that
`(f ∘ splat)(x) = f(x...)`. For example,

    f(x, y) = (x, y)
    g = f ∘ splat ∘ f
    g(x, y) == (x, y) # true
"""
immutable SplattingFunctor; end
const splat = SplattingFunctor()

"""
A `ComposedFunctor` is a (possibly nested) container of two functors (possibly
standard Julia functions) that can apply the functions in succession.
`ComposedFunctor`s are typically constructed via the `∘` (\\circ) operator or
`compose()` function. Generally, `(f ∘ g)(x...) = f(g(x...))`.

The functor structure is ammenable to operations, special optimizations or
planning. For example, you may wish to define the inverse of your functor via
`inv()`. A simple example of optimization is to remove unecessary operation in
`!(!(x::Bool))` by defining:

    (::ComposedFunctor{typeof(!), typeof(!)})(x::Bool) = x

(see also splat::SplattingFunctor)
"""
immutable ComposedFunctor{F1, F2}
    f1::F1
    f2::F2
end

# Apply the composition the usual way.
(f::ComposedFunctor)(x...) = f.f1(f.f2(x...))
(f::ComposedFunctor{T,SplattingFunctor}){T}(x) = f.f1(x...)

show(io::IO, f::ComposedFunctor) = print(io, "($(f.f1) ∘ $(f.f2))")

"""
    compose(a, b)
    a ∘ b

Create a composition of two functions or functor objects, such that
`(f ∘ g)(x) == f(g(x))`. The `∘` symbol can be accessed using
`\\circ`. By default, `ComposedFunctor(f, g)` is returned, but this may be
overriden to implement optimizations and simplifications.

As a special case, the `!` function on functions creates a composition, e.g.
`!isless = ! ∘ isless`, which among other things allows shorthand syntax for
predicate maps of the form `map(!f, iterable)`.

(see also splat::SplattingFunctor)
"""
compose(a, b) = ComposedFunctor(a, b)
compose(a, b::ComposedFunctor{SplattingFunctor}) = ComposedFunctor(ComposedFunctor(a, splat), b.f2) # Make sure splat never appears/stays on left

const ∘ = compose

"""
    !func

Applying `!` to functions creates a function composition `! ∘ func`, which among
other things enables shorthand syntax for predicate maps of the form
`map(!func, iterable)`.
"""
Base.:!(f::Union{Function,ComposedFunctor}) = Base.:! ∘ f

"""
   inv(func)

Return the inverse of a function or functor. For example, `inv(identity) = identity`.
Although this is not defined on many functions, it is an interface available
for overloading functors in packages.
"""
inv(f::ComposedFunctor) = inv(f.f2) ∘ inv(f.f1)
inv(::Type{identity}) = identity

# Some related simplifications for identity
compose(::Type{identity}, f) = f
compose(f, ::Type{identity}) = f
