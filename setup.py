from setuptools import find_packages, setup

setup(
    name="spkemb_ecapa2",
    version="0.0.1",
    description="spkemb_ecapa2",
    author="Kai Washizaki",
    author_email="bandad.kw@gmail.com",
    long_description_content_type="text/markdown",
    package_data={"": ["_example_data/*"]},
    packages=find_packages(include=["src*", "tests*"]),
    include_package_data=True,
)