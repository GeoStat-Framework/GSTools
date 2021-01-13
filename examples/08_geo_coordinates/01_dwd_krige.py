"""
Kriging geographical data
-------------------------

In this example we are going to interpolate actual temperature data from
the German weather service `DWD <https://www.dwd.de/EN>`_.

Data is retrieved utilizing the beautiful package
`wetterdienst <https://github.com/earthobservations/wetterdienst>`_,
which serves as an API for the DWD data.

For better visualization, we also download a simple shapefile of the German
borderline with `cartopy <https://github.com/SciTools/cartopy>`_.

In order to keep the number of dependecies low, the calls of both functions
shown beneath are commented out.
"""
# sphinx_gallery_thumbnail_number = 2
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs


def get_borders_germany():
    """Download simple german shape file with cartopy."""
    from cartopy.io import shapereader as shp_read  # version 0.18.0
    import geopandas as gp  # 0.8.1

    shpfile = shp_read.natural_earth("50m", "cultural", "admin_0_countries")
    df = gp.read_file(shpfile)  # only use the simplest polygon
    poly = df.loc[df["ADMIN"] == "Germany"]["geometry"].values[0][0]
    np.savetxt("de_borders.txt", list(poly.exterior.coords))


def get_dwd_temperature():
    """Get air temperature from german weather stations from 9.6.20 12:00."""
    from wetterdienst.dwd import observations as obs  # version 0.10.1

    sites = obs.DWDObservationSites(
        parameter_set=obs.DWDObservationParameterSet.TEMPERATURE_AIR,
        resolution=obs.DWDObservationResolution.HOURLY,
        period=obs.DWDObservationPeriod.RECENT,
        start_date="2020-06-09 12:00:00",
        end_date="2020-06-09 12:00:00",
    )
    df0 = sites.all()
    ids, lat, lon = map(np.array, [df0.STATION_ID, df0.LAT, df0.LON])
    observations = obs.DWDObservationData(
        station_ids=ids,
        parameters=obs.DWDObservationParameter.HOURLY.TEMPERATURE_AIR_200,
        resolution=obs.DWDObservationResolution.HOURLY,
        start_date="2020-06-09 12:00:00",
        end_date="2020-06-09 12:00:00",
    )
    df1 = observations.collect_safe()
    temp, ids1 = map(np.array, [df1.VALUE, df1.STATION_ID])
    select = np.isfinite(temp)  # care about missing values
    sorter = np.argsort(ids)  # care about missing stations
    sort = sorter[np.searchsorted(ids, ids1[select], sorter=np.argsort(ids))]
    ids, lat, lon, temp = ids[sort], lat[sort], lon[sort], temp[select]
    head = "id, lat, lon, temp"  # add a header to the file
    np.savetxt("temp_obs.txt", np.array([ids, lat, lon, temp]).T, header=head)


###############################################################################
# If you want to download the data again,
# uncomment the two following lines. We will simply load the resulting
# files to gain the border polygon and the observed temperature along with
# the station locations given by lat-lon values.

# get_borders_germany()
# get_dwd_temperature()

border = np.loadtxt("de_borders.txt")
ids, lat, lon, temp = np.loadtxt("temp_obs.txt").T

###############################################################################
# First we will estimate the variogram of our temperature data.
# As the maximal bin distance we choose 8 degrees, which corresponds to a
# chordal length of about 900 km.

bin_max = np.deg2rad(8)
bins = np.linspace(0, bin_max, 20)
bin_c, vario = gs.vario_estimate((lat, lon), temp, bins, latlon=True)

###############################################################################
# Now we can use this estimated variogram to fit a model to it.
# Here we will use a :any:`Spherical` model. We select the ``latlon`` option
# to use the `Yadrenko` variant of the model to gain a valid model for lat-lon
# coordinates and we rescale it to the earth-radius. Otherwise the length
# scale would be given in radians representing the great-circle distance.
#
# We deselect the nugget from fitting and plot the result afterwards.
#
# .. note::
#
#    You need to plot the Yadrenko variogram, since the standard variogram
#    still holds the ordinary routine that is not respecting the great-circle
#    distance.

model = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
model.fit_variogram(bin_c, vario, nugget=False)
ax = model.plot("vario_yadrenko", x_max=bin_max)
ax.scatter(bin_c, vario)
print(model)

###############################################################################
# As we see, we have a rather large correlation length of 600 km.
#
# Now we want to interpolate the data using :any:`Universal` kriging.
# In order to tinker around with the data, we will use a north-south drift
# by assuming a linear correlation with the latitude.
# This can be done as follows:


def north_south_drift(lat, lon):
    return lat


uk = gs.krige.Universal(
    model=model,
    cond_pos=(lat, lon),
    cond_val=temp,
    drift_functions=north_south_drift,
)

###############################################################################
# Now we generate the kriging field, by defining a lat-lon grid that covers
# the whole of Germany. The :any:`Krige` class provides the option to only
# krige the mean field, so one can have a glimpse at the estimated drift.

g_lat = np.arange(47, 56.1, 0.1)
g_lon = np.arange(5, 16.1, 0.1)

field, k_var = uk((g_lat, g_lon), mesh_type="structured")
mean, m_var = uk((g_lat, g_lon), mesh_type="structured", only_mean=True)

###############################################################################
# And that's it. Now let's have a look at the generated field and the input
# data along with the estimated mean:

levels = np.linspace(5, 23, 64)
fig, ax = plt.subplots(1, 3, figsize=[10, 5], sharey=True)
sca = ax[0].scatter(lon, lat, c=temp, vmin=5, vmax=23, cmap="coolwarm")
co1 = ax[1].contourf(g_lon, g_lat, field, levels, cmap="coolwarm")
co2 = ax[2].contourf(g_lon, g_lat, mean, levels, cmap="coolwarm")

[ax[i].plot(border[:, 0], border[:, 1], color="k") for i in range(3)]
[ax[i].set_xlim([5, 16]) for i in range(3)]
[ax[i].set_xlabel("Lon in deg") for i in range(3)]
ax[0].set_ylabel("Lat in deg")

ax[0].set_title("Temperature observations at 2m\nfrom DWD (2020-06-09 12:00)")
ax[1].set_title("Interpolated temperature\nwith North-South drift")
ax[2].set_title("Estimated mean drift\nfrom Universal Kriging")

fmt = dict(orientation="horizontal", shrink=0.5, fraction=0.1, pad=0.2)
fig.colorbar(co2, ax=ax, **fmt).set_label("T in [°C]")

###############################################################################
# To get a better impression of the estimated north-south drift, we'll take
# a look at a cross-section at a longitude of 10 degree:

fig, ax = plt.subplots()
ax.plot(g_lat, field[:, 50], label="Interpolated temperature")
ax.plot(g_lat, mean[:, 50], label="North-South mean drift")
ax.set_xlabel("Lat in deg")
ax.set_ylabel("T in [°C]")
ax.set_title("North-South cross-section at 10°")
ax.legend()

###############################################################################
# Interpretion of the results is now up to you! ;-)
