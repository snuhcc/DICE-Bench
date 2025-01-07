## define your functions
## base: examples

def book_flight(departure: str, destination: str, departure_time: str, return_time: str) -> dict:
    """
    Books a flight given the required details.

    Args:
        departure (str): Departure airport.
        destination (str): Destination airport.
        departure_time (str): Date and time of departure.
        return_time (str): Date and time of return.

    Returns:
        dict: Booking confirmation with flight details.
    """
    pass

def book_accommodation(location: str, check_in: str, check_out: str, guests: int) -> dict:
    """
    Books an accommodation at a specified location.

    Args:
        location (str): Desired location for the accommodation.
        check_in (str): Check-in date and time.
        check_out (str): Check-out date and time.
        guests (int): Number of guests.

    Returns:
        dict: Booking confirmation with accommodation details.
    """
    pass

def book_rental_car(car_type: str, pickup_date: str, return_date: str) -> dict:
    """
    Books a rental car for the specified period.

    Args:
        car_type (str): Type of car to rent (e.g., SUV, sedan).
        pickup_date (str): Date and time for car pickup.
        return_date (str): Date and time for car return.

    Returns:
        dict: Booking confirmation with rental car details.
    """
    pass