package com.example.aiobs.ai

/** Live lifecycle states actually emitted by ObjectTracker. */
enum class TrackState {
    ACTIVE,
    LOST
}

/** Public alias used by the overlay/UI. */
typealias TrackedObject = ObjectTracker.TrackedObject
