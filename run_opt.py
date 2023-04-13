import building_object as bldg

build = bldg.Building(
        typ="SFH",
        age="1984_1994",
        loc="Marienberg",
        apart_quant=1,
        apart_size=81 * 2,
        apart_habitant=3,
        opt=bldg.Options(project_name="First_example", useable_roofarea=0.8)
        )
build.opti_model()
build.run_opti()
build.postprocess_model()