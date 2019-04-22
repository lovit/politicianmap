`News` 는 날짜별 뉴스 (doc) 를 탐색하는 클래스이다.

```python
from politicianmap.utils import News


data_dirname = '/workspace/lovit/politicianmap/tokenized/'
index_dirname = '/workspace/data/politician_norm/'
idx = 0
news = News(
    dirname = '{}/{}/'.format(data_dirname, idx),
    indexdirname = '{}/{}/'.format(index_dirname, idx),
)
```

News class instance 를 만들면 해당 위치의 날짜 및 뉴스 기사의 개수가 확인된다.

```
/workspace/lovit/politicianmap/tokenized//0/ has news of 2224 dates, 223590 docs
```

특정 날짜의 뉴스만 선택하고 싶다면 News 를 만들 때, 날짜를 선택할 수도 있다.

```python
news = News(
    dirname = '{}/{}/'.format(data_dirname, idx),
    indexdirname = '{}/{}/'.format(index_dirname, idx),
    begin_date = '2015-01-01',
    end_date = '2016-01-30'
)
```

해당 날짜의 뉴스만 선택된다.

```
/workspace/lovit/politicianmap/tokenized//0/ has news of 395 dates, 98151 docs
```

특정 기간의 뉴스만 가져오기 위해서는 `get_news` 함수를 이용할 수 있다.
날짜를 입력하면 해당 날짜의 뉴스가 iter 된다.

```python
begin_date = '2016-10-20'
end_date = '2016-10-20'
for doc in news.get_news(begin_date, end_date):
    print(len(doc.split()))
```

```
432
1215
121
286
212
126
121
636
535
```

`DateDocsDecorator` 는 date 와 doc 을 함께 yield 하는 decorator 이다.
`min_doc` 을 통하여 하루의 뉴스가 이 개수보다 작을 때는 해당 날짜는 무시한다.
docs 는 news.get_news() 를 통하여 얻어진 list of str 이다.

```python
news = News(
    dirname = '{}/{}/'.format(data_dirname, idx),
    indexdirname = '{}/{}/'.format(index_dirname, idx),
    begin_date = '2015-01-01',
    end_date = '2016-01-30'
)

date_news = DateDocsDecorator(news, min_doc=10)

for date, docs in date_news:
    # do something
    print(type(docs)) # list
```
