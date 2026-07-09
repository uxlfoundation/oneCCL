===================================
Caching of Communication Operations
===================================

Communication operations may have expensive initialization phase
(for example, allocation of internal structures and buffers, registration of memory buffers, handshake with peers, and so on). 
|product_short| amortizes these overheads by caching operation internal representations and reusing them on the subsequent calls.

To control this, use operation attribute and set ``true`` value for ``to_cache`` field and unique string (for example, tensor name) for ``match_id`` field.

Note that:

- ``match_id`` should be the same for a specific communication operation across all ranks.
- If the same tensor is a part of different communication operations, ``match_id`` should have different values for each of these operations.
