Tutorials
=========

These tutorials walk you through using **Pyriodic** for circular data analysis, from getting started to phase diagnostics and circular statistics.

.. raw:: html

    <div class="row">
      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="01_intro.html" class="stretched-link"></a>
          <img src="../_static/tutorials/01_intro.png" class="card-img-top" alt="Overview of respiration data preprocessing and analysis">
          <div class="card-body">
            <h5 class="card-title">Overview of respiration data preprocessing and analysis</h5>
            <p class="card-text">This tutorial covers a basic pipeline for preprocessing and analysis of respiration data collected during a experimental task. It introduces core `pyriodic` classes.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="02_phase_extraction.html" class="stretched-link"></a>
          <img src="../_static/tutorials/02_phase_extraction.png" class="card-img-top" alt="Methods for phase extraction from respiration data">
          <div class="card-body">
            <h5 class="card-title">Methods for phase extraction from respiration data</h5>
            <p class="card-text">See notebook for details.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="03_descriptive_stats.html" class="stretched-link"></a>
          <img src="../_static/tutorials/03_descriptive_stats.png" class="card-img-top" alt="Descriptive statistics for circular data">
          <div class="card-body">
            <h5 class="card-title">Descriptive statistics for circular data</h5>
            <p class="card-text">We will explore how to compute descriptive statistics for circular data using the Pyriodic package, and illustrate why we need special methods for circular data.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="04a_permutation_against_surrogate_shuffled.html" class="stretched-link"></a>
          <img src="../_static/tutorials/04a_permutation_against_surrogate_shuffled.png" class="card-img-top" alt="Single-level analysis">
          <div class="card-body">
            <h5 class="card-title">Single-level analysis</h5>
            <p class="card-text">See notebook for details.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="05_permutation_two_samples.html" class="stretched-link"></a>
          <img src="../_static/tutorials/05_permutation_two_samples.png" class="card-img-top" alt="Two sample permutation test of respiratory phase angles">
          <div class="card-body">
            <h5 class="card-title">Two sample permutation test of respiratory phase angles</h5>
            <p class="card-text">This notebook runs permutation tests to compare two samples of respiratory phase angles. This is useful when you want to compare the phase angles of two different conditions or groups.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="05_permutation_two_samples_grouplevel.html" class="stretched-link"></a>
          <img src="../_static/tutorials/default.png" class="card-img-top" alt="Two sample permutation test of respiratory phase angles at the group level">
          <div class="card-body">
            <h5 class="card-title">Two sample permutation test of respiratory phase angles at the group level</h5>
            <p class="card-text">This notebook runs permutation tests to compare two samples of respiratory phase angles. This is useful when you want to compare the phase angles of two different conditions or groups.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="07_association_with_behavioural_variable.html" class="stretched-link"></a>
          <img src="../_static/tutorials/07_association_with_behavioural_variable.png" class="card-img-top" alt="Test association with behavioural variable such as reaction time">
          <div class="card-body">
            <h5 class="card-title">Test association with behavioural variable such as reaction time</h5>
            <p class="card-text">This notebook runs permutation tests to see whether there is a statistically significant relationship between the phase angle at which a target stimulus was presented and the response was given by the participant. The analysis is done for each participant separately, and the results are visualised in polar plots.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="08_methods_for_generating_surrogates.html" class="stretched-link"></a>
          <img src="../_static/tutorials/default.png" class="card-img-top" alt="Choosing a method for generating surrogate data for inference">
          <div class="card-body">
            <h5 class="card-title">Choosing a method for generating surrogate data for inference</h5>
            <p class="card-text">See notebook for details.</p>
          </div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card shadow-sm mb-4">
          <a href="IAAFT_draft.html" class="stretched-link"></a>
          <img src="../_static/tutorials/IAAFT_draft.png" class="card-img-top" alt="Computation times">
          <div class="card-body">
            <h5 class="card-title">Computation times</h5>
            <p class="card-text">See notebook for details.</p>
          </div>
        </div>
      </div>

    </div>

.. include:: _image_manifest.rst

.. toctree::
   :hidden:

   01_intro
   02_phase_extraction
   03_descriptive_stats
   04a_permutation_against_null_timeshifted_DRAFT
   04a_permutation_against_surrogate_shuffled
   04a_permutation_against_surrogate_shuffled_grouplevel_DRAFT
   05_permutation_two_samples
   05_permutation_two_samples_grouplevel
   06_why_circular_stats_DRAFT
   07_association_with_behavioural_variable
   08_methods_for_generating_surrogates
   09_physiological_parameters_DRAFT
   100_response_DRAFT
   IAAFT_draft
   testing_DRAFT
