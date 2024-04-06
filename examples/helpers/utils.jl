function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function generate_almost_grid(
    N_on_grid,
    N_randomly_generated,
    lb::T,
    ub::T,
    ) where T

    #
    N_grid = N_on_grid
    N_rand = N_randomly_generated

    #N_grid = 523 # non-posdef kernel matrix.
    #N_rand = 54 # non-posdef kernel matrix.
    N = N_grid + N_rand
    
    # legacy comment: if 0 and 1 are included, we have posdef error, or rank = N - 2.
    x_range = LinRange(lb, ub, N-N_rand)
    x_rand = collect(
        convertcompactdomain(
            rand(T),
            zero(T), one(T),
            x_range[1], x_range[end],
        )
        for _ = 1:N_rand
    )
    x_all = sort(vcat(x_range, x_rand))
    
    X = collect( x_all[n] for n = 1:N )

    return X
end

# Central Park Towers
function get_new_york_station()
    return "USW00094728"
end

# University.
function get_seattle_station()
    return "USC00457478"
end

# Toronto Island A.
function get_toronto_station()
    return "CA006158665"
end

function load_data(station_name)
    data_path = joinpath("data", "$(station_name).csv")
    return CSV.read(data_path, DF.DataFrame)
end