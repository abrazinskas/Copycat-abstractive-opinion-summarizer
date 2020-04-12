from .ordered_attrs import OrderedAttrs
from mltoolkit.mlutils.helpers.formatting.general import format_title, \
    format_signature
from mltoolkit.mlutils.tools.signature_scraper import SignatureScraper


class SignedObject(OrderedAttrs):
    """
    Objects of this class have an internal scrapping mechanism in order
    to produce user-friendly prints based on attributes scraping and formatting.
    """

    def __init__(self, name_prefix=None):
        """
        :param name_prefix: a str that will prefix the title of the object if
                            the signature is generated.
        """
        super(SignedObject, self).__init__()
        self.name_prefix = name_prefix

        self.scraper = SignatureScraper(excl_attr_names=['name_prefix',
                                                         'scraper'],
                                        scrape_obj_vals=True)

    def get_title(self):
        """Returns the formatted title of the object with a prefix if set."""
        cl_name = self.__class__.__name__
        return format_title(cl_name, name_prefix=self.name_prefix)

    def get_sign_attrs(self):
        """Returns attrs (names and values) defining the object."""
        return self.scraper.scrape(self)

    def __repr__(self):
        attrs = self.get_sign_attrs()
        attrs['_TITLE_'] = self.get_title()
        return attrs

    def __str__(self):
        """Converts the object's conf/setup into a human readable string."""
        title = self.get_title()
        attrs = self.get_sign_attrs()
        return format_signature(title, attrs, indent=2)
