# YouTubeNotebooks
Video: [Automatic Differentiation (AutoDiff) with Julia Video on YouTube](https://www.youtube.com/watch?v=vAp6nUMrKYg)

To run the notebook the easiest is to go to [juliabox.com](http://juliabox.com).

To just see the notebook: http://nbviewer.jupyter.org/github/alanedelman/YouTubeNotebooks/blob/master/Automatic%20Differentiation%20in%2010%20Minutes.ipynb


struct D <: Number  # D is a function-derivative pair
    f::Tuple{Float64,Float64}
end

import Base: +, /, convert, promote_rule
+(x::D, y::D) = D(x.f .+ y.f)
/(x::D, y::D) = D((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
convert(::Type{D}, x::Real) = D((x,zero(x)))  # derivative of a constant
promote_rule(::Type{D}, ::Type{<:Number}) = D


struct D1{T} <: Number  # D is a function-derivative pair
    f::Tuple{T,T}
end

import Base: +, /, convert, promote_rule
+(x::D1, y::D1) = D1(x.f .+ y.f)
/(x::D1, y::D1) = D1((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
convert(::Type{D1{T}}, x::Real) where {T} = D1((convert(T, x), zero(T)))
promote_rule(::Type{D1{T}}, ::Type{S}) where {T,S<:Number} = D1{promote_type(T,S)}
