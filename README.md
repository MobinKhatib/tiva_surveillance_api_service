## 📌 Project Overview

These repositories showcase two examples of our work, each designed to operate in different deployment scenarios.

At our company, we work with a specialized camera system that captures individual frames — each containing exactly one vehicle. The workflow is structured as follows:::

- **Client Integration:**
  - Clients send data to our system via API.
  - The payload is formatted in JSON and typically includes:
    - `original_timestamp`
    - A base64-encoded image frame (one car per frame)
    - Optional metadata (e.g., `location`, `street_name`, etc.)

- **Processing Pipeline:**
  - Detect the vehicle in the image.
  - Identify the license plate.
  - Extract characters from the plate.

- **Response:**
  - We send back a JSON object containing:
    - `status`
    - `timestamp`
    - `plate` (extracted characters)
    - A base64-encoded processed image (e.g., with annotations)
    - Any additional metadata if provided in the request
