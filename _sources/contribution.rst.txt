Contribution Guide
=========================

Contributions Welcome!
-------------------------

This OSS library is a personal project and hasn't received any corporate support.
Therefore, development usually takes place during the night.
If you find this library interesting, please feel free to use it and share your thoughts.
If you have questions, suggestions, bug reports, or feature requests, don't hesitate to post them in Issues or Discussions.

Since it's challenging to gauge the current demand, we have not set any specific format for contributions.

.. note::

    To eliminate language barriers, you are free to write issues in any language.
    However, please note that since the current main maintainers can only read English and Japanese,
    responses to other languages will all be handled through machine translation.

    言語による障壁をなくすため、基本的にどの言語でissueを書いてもいいですが、
    現在の主メンテナは英語と日本語しか読めないので他の言語はすべて機械翻訳での応対になることに注意してください。

    为了消除语言障碍，您基本上可以用任何语言来写问题，但是请注意，由于当前的主要维护者只能阅读英语和日语，因此其他语言的回应都将通过机器翻译进行。

    언어에 의한 장벽을 없애기 위해 기본적으로 어떤 언어로든지 이슈를 작성하실 수 있지만, 
    현재 주요 관리자는 영어와 일본어만 읽을 수 있으므로 다른 언어는 모두 기계 번역으로 응대될 것임을 유의해 주세요.

Movis.contrib Namespace
--------------------------

Modifications that maintain backward compatibility or clearly enhance convenience can be directly added to the ``movis`` namespace.
However, when adding experimental modules, it is preferable to submit a PR to the ``movis.contrib`` namespace.
Within this namespace, even if libraries not used by vovis are added,
they will be merged with minimal testing and documentation as long as they are not directly imported.

If modules in this namespace gain widespread use, we may consider incorporating them directly into the ``movis`` namespace in the future.

.. note::
   Areas actively merged:

   - Bug fixes
   - Feature additions that do not break backward compatibility
   - Addition of documentation
   - Feature additions related to tutorial and presentation videos, especially if deemed in high demand

     - For example: Auto-generation of assets, addition of transition effects, etc.
     - In particular, features related to subtitles are expected to be in high demand.

   - Addition of examples


.. warning::
   Areas not actively merged:

   - Changes to protocols
   - Addition of functions without Docstring
   - Feature additions that add dependencies (However, this does not apply to the ``movis.contrib`` namespace)
