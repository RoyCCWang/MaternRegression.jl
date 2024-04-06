
function moveall(
    query_string,
    file_names,
    dest_dir;
    force = true)

    f = xx->occursin(query_string, xx)
    out = filter(f, file_names)

    for s in out

        mv(
            s,
            joinpath(dest_dir, s);
            force = force,
        )
    end
    
    return nothing
end

function reset_dir(dest_dir)
    if ispath(dest_dir)
        t = @task begin
            rm(dest_dir; recursive = true)
        end
        schedule(t)
        wait(t)
    end
    
    if !ispath(dest_dir)
        mkpath(dest_dir)
    end
end

function move_md(dest_dir, move_prefix_name)

    t = @task begin
        file_names = readdir()
        moveall(
            move_prefix_name,
            file_names,
            dest_dir;
        )
    end
    schedule(t)
    wait(t)
    
    return nothing
end