using XCALibre, StaticArrays, LinearAlgebra

"""Fixed-domain independently segmented multilayer polygonal annuli. Only the
shared radius-one interface is interpreted as circular; exterior walls are chords."""
function circular_annulus_mesh(;na=16,nb=19,nra=3,nrb=3,phase=0.13,
    inner_radius=0.5,outer_radius=1.5)
    P=XCALibre.Point
    points=typeof(P(0.,0.,0.))[]
    edges=typeof(line!([P(0.,0.,0.),P(1.,0.,0.)],1,2,1))[]
    blocks=XCALibre.Block{Int64}[]
    ia,ib,inner,outer=Int[],Int[],Int[],Int[]
    for (side,n,nr,shift,lo,hi) in ((1,na,nra,0.,inner_radius,1.),(2,nb,nrb,phase,1.,outer_radius))
        ts=[shift+2pi*k/n for k in 0:n-1];offset=length(points);base=length(edges)
        for r in range(lo,hi;length=nr+1),t in ts
            push!(points,P(r*cos(t),r*sin(t),0.))
        end
        for layer in 0:nr,k in 1:n
            push!(edges,line!(points,offset+layer*n+k,offset+layer*n+mod1(k+1,n),1))
        end
        radialbase=length(edges)
        for layer in 0:nr-1,k in 1:n
            push!(edges,line!(points,offset+layer*n+k,offset+(layer+1)*n+k,1))
        end
        for layer in 0:nr-1,k in 1:n
            push!(blocks,quad(edges,[base+layer*n+k,base+(layer+1)*n+k,
                radialbase+layer*n+k,radialbase+layer*n+mod1(k+1,n)]))
        end
        if side==1
            append!(ia,base+nr*n .+ (1:n));append!(inner,base .+ (1:n))
        else
            append!(ib,base .+ (1:n));append!(outer,base+nr*n .+ (1:n))
        end
    end
    patches=[Patch(:interface_a,ia),Patch(:interface_b,ib),Patch(:inner,inner),Patch(:outer,outer)]
    XCALibre.UNV2.update_mesh_format(generate!(MeshBuilder2D(points,edges,patches,blocks)),Int64,Float64)
end

function circular_couette_case(;na=16,nb=19,nra=3,nrb=3,phase=0.13,omega=0.1,nu=1.,
    relaxation_U=0.7,relaxation_p=0.3,initial_pressure=0.,reference_pressure=0.,
    wall_pressure=:homogeneous,kwargs...)
    mesh=circular_annulus_mesh(;na,nb,nra,nrb,phase)
    c=construct_circular_interface(mesh,:interface_a,:interface_b;centre=(0.,0.,0.),radius=1.)
    r=construct_circular_reconstruction(mesh,c)
    wall_pressure in (:homogeneous,:analytical) || throw(ArgumentError("unsupported wall pressure mode"))
    kind=wall_pressure==:analytical ? :normal_gradient : :neumann
    h=construct_circular_pressure_geometry(mesh,c,fill(kind,length(r.exterior_faces)))
    m=construct_circular_momentum_geometry(h,fill(:dirichlet,length(r.exterior_faces)))
    a,b=0.5,1.5;A=-omega*a^2/(b^2-a^2);B=omega*a^2*b^2/(b^2-a^2)
    exact_velocity(x)=begin
        radius=norm(x);speed=A*radius+B/radius
        speed*SVector(-x[2]/radius,x[1]/radius,0.)
    end
    primitive(radius)=A^2*radius^2/2+2A*B*log(radius)-B^2/(2radius^2)
    values=[norm(x)<1 ? omega*SVector(-x[2],x[1],0.) : SVector(0.,0.,0.) for x in r.exterior_points]
    pressure_values=[wall_pressure==:homogeneous ? 0. :
        (A*norm(x)+B/norm(x))^2/norm(x)*dot(x/norm(x),normalize(r.metrics.face_area_vector[j]))
        for (x,j) in zip(r.exterior_points,r.exterior_faces)]
    exact_gradient(x)=(A*norm(x)+B/norm(x))^2/norm(x)*(x/norm(x))
    state=CircularSimpleState(m,fill(SVector(0.,0.,0.),length(r.points)),fill(initial_pressure,length(r.points)),
        values,pressure_values,nu;reference=Dict(1=>reference_pressure),relaxation_U,relaxation_p,kwargs...)
    (;mesh,coupling=c,state,exact_velocity,primitive,exact_gradient,wall_pressure)
end

function circular_couette_errors(case)
    s=case.state;r=s.geometry.pressure.reconstruction
    exact=case.exact_velocity.(r.points)
    velocity=sqrt(sum(r.metrics.volume.*norm.(s.velocity-exact).^2)/sum(r.metrics.volume.*norm.(exact).^2))
    pressure_exact=[case.primitive(norm(x))-case.primitive(norm(r.points[1]))+s.config.reference[1] for x in r.points]
    pressure=maximum(abs,s.pressure-pressure_exact)
    gradient=fill(SVector(0.,0.,0.),length(s.pressure))
    circular_pressure_gradient!(gradient,s.geometry.pressure,s.pressure,s.config.pressure_values)
    pressure_gradient=maximum(norm,gradient-case.exact_gradient.(r.points))
    (;velocity,pressure,pressure_gradient)
end
