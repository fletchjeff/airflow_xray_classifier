FROM quay.io/astronomer/astro-runtime:5.0.5

COPY --chown=astro:astro include/.aws /home/astro/.aws