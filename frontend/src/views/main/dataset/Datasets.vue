<template>
  <div>
    <v-toolbar light>
      <v-toolbar-title>
        Manage Datasets
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn color="primary" to="/main/datasets/create">Create Dataset</v-btn>
    </v-toolbar>
    <v-data-table :headers="headers" :items="datasets" item-key="name">
      <template v-slot:item="{ item }">
        <tr>
          <td>{{ item.title }}</td>
          <td>
            <v-btn text :to="{name: 'main-datasets-edit', params: {id: item.id}}">
              <v-icon>edit</v-icon>
            </v-btn>
          </td>
        </tr>
      </template>
    </v-data-table>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Store } from 'vuex';
import { Dataset } from '@/interfaces';
import { readDatasets } from '@/store/dataset/getters';
import { dispatchGetDatasets } from '@/store/dataset/actions';

@Component
export default class DatasetDatasets extends Vue {
  public headers = [
    {
      text: 'Title',
      sortable: true,
      value: 'title',
      align: 'left',
    },
    {
      text: 'Actions',
      value: 'id',
    },
  ];
  get datasets() {
    return readDatasets(this.$store);
  }

  public async mounted() {
    await dispatchGetDatasets(this.$store);
  }
}
</script>
